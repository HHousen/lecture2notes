import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from tqdm import tqdm

from custom_nnmodules import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Slide Classifier Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet34)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=6, type=int, metavar='N',
                    help='number of total epochs to run (default: 6)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=4e-3, type=float, # 1e-4 for whole model # 3e-4 is the best learning rate for Adam, hands down
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-2)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=-1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--random_split', dest='use_random_split', action='store_true',
                    help='use random_split to create train and val set instead of train and val folders')
parser.add_argument('--feature_extract', choices=["normal", "advanced"],
                    help='If False, we finetune the whole model. When normal we only update the reshaped layer params. When advanced use fastai version of feature extracting (add fancy group of layers and only update this group and BatchNorm)')
parser.add_argument('--find_lr', dest='use_find_lr', action='store_true',
                    help='Flag for lr_finder.')
parser.add_argument('-o', '--optimizer', default='adamw', dest='optim',
                    help='Optimizer to use (default=AdamW)')

best_acc1 = 0

def set_parameter_requires_grad(model, feature_extracting):
    """This helper function sets the .requires_grad attribute of the parameters in 
    the model to False when we are feature extracting. By default, when we load a 
    pretrained model all of the parameters have .requires_grad=True, which is fine 
    if we are training from scratch or finetuning. However, if we are feature 
    extracting and only want to compute gradients for the newly initialized layer 
    then we want all of the other parameters to not require gradients."""
    if feature_extracting:
        for module in model.modules():
            for param in module.parameters():
                # Don't set BatchNorm layers to .requires_grad False because that's what fastai does
                # but only if using feature_extracting == "advanced"
                if feature_extracting == "advanced" and isinstance(module, nn.BatchNorm2d):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

def unfreeze_model(model):
    """Sets every layer of model to trainable (.requires_grad True)"""
    for param in model.parameters():
        param.requires_grad = True

def main():
    
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU.')

    ngpus_per_node = torch.cuda.device_count()
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    train_dataset, val_dataset, classes = get_datasets(args)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Create model
    model, input_size, params_to_update = initialize_model(len(classes), args.feature_extract, args)
    print("Model:\n" + str(model))

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.use_find_lr:
        from lr_finder import LRFinder
        lr_optimizer = torch.optim.Adam(params_to_update, lr=1e-7, weight_decay=1e-2)
        lr_finder = LRFinder(model, lr_optimizer, criterion, device="cuda")
        lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
        lr_finder.plot()
        exit()

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(params_to_update, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params_to_update, args.lr,
                                weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr, 
        steps_per_epoch=len(train_loader), 
        epochs=args.epochs)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Overall"):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'model': model,
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'class_index': classes
        }, is_best)

def get_datasets(args):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if args.use_random_split:
        # Random split dataset
        full_dataset = datasets.ImageFolder(
            args.data,
            transforms.Compose([
                transforms.Resize((480,640)),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        classes = full_dataset.classes
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )
        val_dataset = datasets.ImageFolder(
            valdir, 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        )
        classes = train_dataset.classes

    # print(full_dataset.class_to_idx)
    return train_dataset, val_dataset, classes

def create_head(out_features):
    layers = [
        AdaptiveConcatPool2d(1),
        Flatten(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.25),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, out_features)
    ]
    return nn.Sequential(*layers)

def initialize_model(num_classes, feature_extract, args):
    def get_updateable_params(model, feature_extract):
        """
        Gather the parameters to be optimized/updated in this run
        If feature_extract is normal then params should only be fully connected layer
        If feature_extract is advanced then params should be all BatchNorm layers and head (create_head) of network
        If feature_extract is None then params is model.named_parameters()
        """
        params_to_update = model.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            params_to_update = model_ft.parameters()
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
        return params_to_update
    
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = models.__dict__[args.arch](pretrained=args.pretrained)
    input_size = 0
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch.startswith("resnet"):
        """ Resnet """
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features

        # TODO: Implement advanced pretraining technique from fastai for all models
        if feature_extract == "advanced":
            # Use advanced pretraining technique from fastai - https://docs.fast.ai/vision.learner.html#cnn_learner
            # Remove last two layers
            model_ft = nn.Sequential(*list(model_ft.children())[:-2])
            # append head to resnet body
            model_ft = nn.Sequential(model_ft, create_head(num_classes))
        else:
            # Reshape model so last layer has 512 input features and num_classes output features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif args.arch.startswith("alexnet"):
        """ Alexnet """
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif args.arch.startswith("vgg"):
        """ VGG11_bn """
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif args.arch.startswith("squeezenet"):
        """ Squeezenet """
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif args.arch.startswith("densenet"):
        """ Densenet """
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif args.arch.startswith("inception"):
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    params_to_update = get_updateable_params(model_ft, feature_extract)

    return model_ft, input_size, params_to_update

def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    num_batches = len(train_loader)
    batch_time = AverageMeter('Time', ':5.3f')
    data_time = AverageMeter('Data', ':5.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':5.3f')
    progress = ProgressMeter(
        num_batches,
        [batch_time, data_time, losses, top1],
        prefix="Train Epoch [{}]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in tqdm(enumerate(train_loader), total=num_batches, desc=("Train Epoch [" + str(epoch) + "]")):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.optim == "adamw":
            # Implementing AdamW - https://www.fast.ai/2018/07/02/adam-weight-decay/
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-args.weight_decay * group['lr'], param.data)
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.print_freq != -1 and i % args.print_freq == 0:
            progress.display(i)
    progress.display(average=True)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':5.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test'
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm(enumerate(val_loader), total=len(val_loader), desc=("Test")):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.print_freq != -1 and i % args.print_freq == 0:
                progress.display(i)

        progress.display(average=True)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def avg_string(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch=0, average=False):
        if average:
            entries = ["\n" + self.prefix + " Average: "]
            entries += [meter.avg_string() for meter in self.meters]
            entries += "\n"
        else:
            entries = [self.prefix + ": " + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
