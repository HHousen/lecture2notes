import argparse
import logging
import os
import random
import shutil
import sys
import time
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from custom_nnmodules import *  # noqa: F403
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)

if os.environ.get("DISPLAY", "") == "":
    logging.debug("=> MatPlotLib: No display found. Using non-interactive Agg backend")
    mpl.use("Agg")

MODEL_NAMES = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
# Add EfficientNet models separately because they come from the
# lukemelas/EfficientNet-PyTorch package not the main pytorch code
MODEL_NAMES += ["efficientnet-b" + str(i) for i in range(0, 7)]

parser = argparse.ArgumentParser(description="PyTorch Slide Classifier Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet34",
    choices=MODEL_NAMES,
    help="model architecture: " + " | ".join(MODEL_NAMES) + " (default: resnet34)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs",
    default=6,
    type=int,
    metavar="N",
    help="number of total epochs to run (default: 6)",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=16,
    type=int,
    metavar="N",
    help="mini-batch size (default: 16)",
)
# 1e-4 for whole model # 3e-4 is the best learning rate for Adam, hands down
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=4e-3,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--momentum",
    default=0.9,
    type=float,
    metavar="M",
    help="momentum. Ranger optimizer suggests 0.95.",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-2,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-2)",
    dest="weight_decay",
)
parser.add_argument(
    "-k",
    "--ranger-k",
    default=6,
    type=int,
    metavar="K",
    help="Ranger (LookAhead) optimizer k value (default: 6)",
    dest="optim_k",
)
parser.add_argument(
    "--alpha",
    default=0.999,
    type=float,
    metavar="N",
    help="Optimizer alpha parameter (default: 0.999)",
    dest="optim_alpha",
)
parser.add_argument(
    "--eps",
    default=1e-8,
    type=float,
    metavar="N",
    help="Optimizer eps parameter (default: 1e-8)",
    dest="optim_eps",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=-1,
    type=int,
    metavar="N",
    help="print frequency (default: -1)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set and generate overall "
    + "statistics/confusion matrix",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--random_split",
    dest="use_random_split",
    action="store_true",
    help="use random_split to create train and val set instead of train and val folders",
)
parser.add_argument(
    "--relu_to_mish",
    dest="relu_to_mish",
    action="store_true",
    help="convert any relu activations to mish activations",
)
parser.add_argument(
    "--feature_extract",
    choices=["normal", "advanced"],
    help="If False, we finetune the whole model. When normal we only update the reshaped layer params. When advanced use fastai version of feature extracting (add fancy group of layers and only update this group and BatchNorm)",
)
parser.add_argument(
    "--find_lr", dest="use_find_lr", action="store_true", help="Flag for lr_finder."
)
parser.add_argument(
    "-o",
    "--optimizer",
    default="adamw",
    dest="optim",
    help="Optimizer to use (default=AdamW)",
)
parser.add_argument(
    "--tensorboard-model",
    dest="tensorboard_model",
    action="store_true",
    help="Flag to write the model to tensorboard. Action is RAM intensive.",
)
parser.add_argument(
    "--tensorboard",
    default="",
    type=str,
    metavar="PATH",
    help="Path to tensorboard logdir. Tensorboard not used if not set.",
)
parser.add_argument(
    "-l",
    "--log",
    dest="logLevel",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level (default: 'Info').",
)

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
                if feature_extracting == "advanced" and isinstance(
                    module, nn.BatchNorm2d
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False


def unfreeze_model(model):
    """Sets every layer of model to trainable (.requires_grad True)"""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parser.parse_args()

    if args.tensorboard:
        writer = SummaryWriter(args.tensorboard)
    else:
        writer = None

    logging.basicConfig(
        format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
        level=logging.getLevelName(args.logLevel),
    )

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn("You have chosen a specific GPU.")

    ngpus_per_node = torch.cuda.device_count()
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args, writer)


def main_worker(gpu, ngpus_per_node, args, writer=None):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logging.info("Use GPU: {} for training".format(args.gpu))

    input_size = get_input_size(args.arch)
    train_dataset, val_dataset, classes = get_datasets(args, input_size)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Create model
    model, params_to_update = initialize_model(len(classes), args.feature_extract, args)
    logging.debug("Model:\n%s", model)
    logging.debug(f"The model has {count_parameters(model):,} trainable parameters")

    if args.tensorboard and args.tensorboard_model:
        images, _ = next(iter(train_loader))
        writer.add_graph(model, images)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
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
        sys.exit()

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(
            params_to_update,
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "ranger":
        from ranger.ranger import Ranger

        optimizer = Ranger(
            params_to_update,
            args.lr,
            k=args.optim_k,
            betas=(args.momentum, args.optim_alpha),
            eps=args.optim_eps,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            params_to_update,
            args.lr,
            betas=(args.momentum, args.optim_alpha),
            eps=args.optim_eps,
            weight_decay=args.weight_decay,
        )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.error("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs
    )

    if args.evaluate:
        validate(val_loader, model, criterion, writer, args)
        final_evaluate(val_loader, model, classes, args)
        return

    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="Overall"):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch, writer, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, writer, args, epoch=epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(
            {
                "model": model,
                "input_size": input_size,
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
                "class_index": classes,
            },
            is_best,
        )


def get_datasets(args, input_size):
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if args.use_random_split:
        # Random split dataset
        full_dataset = datasets.ImageFolder(
            args.data,
            transforms.Compose(
                [
                    transforms.Resize((480, 640)),
                    transforms.CenterCrop(input_size),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        classes = full_dataset.classes
    else:
        traindir = os.path.join(args.data, "train")
        valdir = os.path.join(args.data, "val")
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.Resize((480, 640)),
                    transforms.CenterCrop(input_size),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize((480, 640)),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        classes = train_dataset.classes

    # print(full_dataset.class_to_idx)
    return train_dataset, val_dataset, classes


def create_head(out_features):
    layers = [
        AdaptiveConcatPool2d(1),  # noqa: F405
        nn.Flatten(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.25),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, out_features),
    ]
    return nn.Sequential(*layers)


def get_input_size(arch):
    if arch.startswith("efficientnet"):
        model_type = arch.split("-", 1)[-1]  # get everything after "-"
        if model_type.startswith("b"):
            input_sizes = [224, 240, 260, 300, 380, 456, 528, 600, 672]
            input_size = input_sizes[
                int(model_type[-1:])
            ]  # select size from array that matches last character of `model_type`
    elif arch.startswith("inception"):
        input_size = 299
    else:
        input_size = 224
    return input_size


def initialize_model(num_classes, feature_extract, args):
    def get_updateable_params(model, feature_extract):
        """
        Gather the parameters to be optimized/updated in this run
        If feature_extract is normal then params should only be fully connected layer
        If feature_extract is advanced then params should be all BatchNorm layers and head (create_head) of network
        If feature_extract is None then params is model.named_parameters()
        """
        params_to_update = model.parameters()
        logging.debug("Params to learn:\n")
        if feature_extract:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    logging.debug("\t%s", name)
        else:
            params_to_update = model_ft.parameters()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    logging.debug("\t%s", name)
        return params_to_update

    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific. EfficientNet is separate because it has not been
    # implemented in the main pytorch repository yet. Instead this script uses the
    # lukemelas/EfficientNet-PyTorch package.
    if args.arch.startswith("efficientnet"):
        if args.pretrained:
            model_ft = EfficientNet.from_pretrained(args.arch)
        else:
            model_ft = EfficientNet.from_name(args.arch)
    else:
        model_ft = models.__dict__[args.arch](pretrained=args.pretrained)

    if args.pretrained:
        logging.info("=> using pre-trained model '{}'".format(args.arch))
    else:
        logging.info("=> creating model '{}'".format(args.arch))

    if args.arch.startswith("resnet"):
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

    elif args.arch.startswith("efficientnet"):
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features  # pylint: disable=protected-access
        model_ft._fc = nn.Linear(
            num_ftrs, num_classes
        )  # pylint: disable=protected-access

    elif args.arch.startswith("alexnet"):
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif args.arch.startswith("vgg"):
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif args.arch.startswith("squeezenet"):
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model_ft.num_classes = num_classes

    elif args.arch.startswith("densenet"):
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif args.arch.startswith("inception"):
        # Be careful, expects (299,299) sized images and has auxiliary output
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxiliary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    else:
        logging.critical("Invalid model name, exiting...")
        sys.exit()

    if args.relu_to_mish:
        convert_relu_to_mish(model_ft)

    params_to_update = get_updateable_params(model_ft, feature_extract)

    return model_ft, params_to_update


def convert_relu_to_mish(model):
    from mish import mish

    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, mish(inplace=True))
        else:
            convert_relu_to_mish(child)


def train(train_loader, model, criterion, optimizer, scheduler, epoch, writer, args):
    num_batches = len(train_loader)
    batch_time = AverageMeter("Batch Time", ":5.3f")
    data_time = AverageMeter("Data Loading Time", ":5.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":5.3f")
    precision_metric = AverageMeter("Precision", ":5.3f")
    recall_metric = AverageMeter("Recall", ":5.3f")
    f_score_metric = AverageMeter("F-Score", ":5.3f")
    progress = ProgressMeter(
        num_batches,
        [
            batch_time,
            data_time,
            losses,
            top1,
            precision_metric,
            recall_metric,
            f_score_metric,
        ],
        prefix="Train Epoch [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in tqdm(
        enumerate(train_loader),
        total=num_batches,
        desc=("Train Epoch [" + str(epoch) + "]"),
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))

        # measure precision, recall, F-measure and support
        precision, recall, f_score, _ = calculate_precision_recall_fscore_support(
            output, target
        )
        precision_metric.update(precision)
        recall_metric.update(recall)
        f_score_metric.update(f_score)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.optim == "adamw":
            # Implementing AdamW - https://www.fast.ai/2018/07/02/adam-weight-decay/
            for group in optimizer.param_groups:
                for param in group["params"]:
                    param.data = param.data.add(
                        -args.weight_decay * group["lr"], param.data
                    )
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.print_freq != -1 and i % args.print_freq == 0:
            progress.display(i)
    progress.display(average=True)

    if args.tensorboard:
        progress.to_tensorboard(epoch, writer, "train")


def validate(val_loader, model, criterion, writer, args, epoch=None):
    batch_time = AverageMeter("Batch Time", ":5.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":5.3f")
    precision_metric = AverageMeter("Precision", ":5.3f")
    recall_metric = AverageMeter("Recall", ":5.3f")
    f_score_metric = AverageMeter("F-Score", ":5.3f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, precision_metric, recall_metric, f_score_metric],
        prefix="Test",
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm(
            enumerate(val_loader), total=len(val_loader), desc="Test"
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))

            # measure precision, recall, F-measure and support
            precision, recall, f_score, _ = calculate_precision_recall_fscore_support(
                output, target
            )
            precision_metric.update(precision)
            recall_metric.update(recall)
            f_score_metric.update(f_score)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.print_freq != -1 and i % args.print_freq == 0:
                progress.display(i)

        progress.display(average=True)

        if args.tensorboard:
            progress.to_tensorboard(epoch, writer, "validate")

    return top1.avg


def final_evaluate(val_loader, model, classes, args):
    # switch to evaluate mode
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for (images, target) in tqdm(
            val_loader, total=len(val_loader), desc=("Final Stats")
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            targets = np.concatenate((targets, target.numpy()))

            # compute output
            output = model(images)
            preds = np.concatenate((preds, torch.argmax(output, 1).cpu().numpy()))

    # print("Preds: " + str(preds))
    # print("Targets: " + str(targets))
    plot_confusion_matrix(preds, targets, classes)
    logging.info(classification_report(targets, preds, target_names=classes))


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.val = None
        self.avg = None
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
        fmtstr = "{name} {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)

    def to_tensorboard(self, y_axis, writer, extra_title):
        writer.add_scalar(self.name + "/" + extra_title, self.avg, y_axis)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
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
        logging.info("\t".join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def to_tensorboard(self, y_axis, writer, extra_title=""):
        for meter in self.meters:
            meter.to_tensorboard(y_axis, writer, extra_title=extra_title)


def plot_confusion_matrix(
    y_pred,
    y_true,
    classes,
    normalize=False,
    title="Confusion Matrix",
    cmap=plt.cm.Blues,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()

    if mpl.backends.backend == "agg":
        plt.savefig("confusion_matrix.png")
    else:
        plt.show()

    return ax


def calculate_precision_recall_fscore_support(output, target):
    with torch.no_grad():
        _, preds = torch.max(output.data, 1)
        precision, recall, f_score, support = precision_recall_fscore_support(
            target.cpu(), preds.cpu(), average="weighted"
        )
        return precision, recall, f_score, support


def accuracy(output, target):
    """Computes the accuracy over the top predictions"""
    with torch.no_grad():
        _, preds = torch.max(output.data, 1)
        acc1 = accuracy_score(target.cpu(), preds.cpu(), normalize=True)
        return acc1


if __name__ == "__main__":
    main()
