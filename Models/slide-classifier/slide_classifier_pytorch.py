import argparse
import os
import random
import logging
from collections import OrderedDict
from tqdm import tqdm

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
from torch.utils.tensorboard import SummaryWriter

from efficientnet_pytorch import EfficientNet
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning import Trainer, seed_everything

from custom_nnmodules import *  # pylint: disable=wildcard-import
from slide_classifier_helpers import convert_relu_to_mish, plot_confusion_matrix

logger = logging.getLogger(__name__)

MODEL_NAMES = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
# Add EfficientNet models separately because they come from the
# lukemelas/EfficientNet-PyTorch package not the main pytorch code
MODEL_NAMES += ["efficientnet-b" + str(i) for i in range(0, 7)]


class SlideClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super(SlideClassifier, self).__init__()

        self.hparams = hparams

        self.classification_model = None
        # If `hparams` has `num_classes` then create the classification model right away.
        # This is necessary for inference. The `num_classes` attribute is saved on model initialization.
        if hasattr(hparams, "num_classes"):
            self.classification_model = self.initialize_model(hparams.num_classes)

        self.hparams.input_size = self.get_input_size()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, *args, **kwargs):
        return self.classification_model(*args, **kwargs)

    def set_parameter_requires_grad(self, model):
        """This helper function sets the .requires_grad attribute of the parameters in
        the model to False when we are feature extracting. By default, when we load a
        pretrained model all of the parameters have .requires_grad=True, which is fine
        if we are training from scratch or finetuning. However, if we are feature
        extracting and only want to compute gradients for the newly initialized layer
        then we want all of the other parameters to not require gradients."""
        if self.hparams.feature_extract and self.hparams.feature_extract != "none":
            for module in model.modules():
                for param in module.parameters():
                    # Don't set BatchNorm layers to .requires_grad False because that's what fastai does
                    # but only if using feature_extract == "advanced"
                    if self.hparams.feature_extract == "advanced" and isinstance(
                        module, nn.BatchNorm2d
                    ):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

    def create_head(self, out_features):
        layers = [
            AdaptiveConcatPool2d(1),
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

    def initialize_model(self, num_classes):
        # Initialize these variables which will be set in this if statement. Each of these
        # variables is model specific. EfficientNet is separate because it has not been
        # implemented in the main pytorch repository yet. Instead this script uses the
        # lukemelas/EfficientNet-PyTorch package.
        if self.hparams.arch.startswith("efficientnet"):
            if self.hparams.pretrained:
                model_ft = EfficientNet.from_pretrained(self.hparams.arch)
            else:
                model_ft = EfficientNet.from_name(self.hparams.arch)
        else:
            model_ft = models.__dict__[self.hparams.arch](
                pretrained=self.hparams.pretrained
            )

        if self.hparams.pretrained:
            logger.info("Using pre-trained model '{}'".format(self.hparams.arch))
        else:
            logger.info("Creating model '{}'".format(self.hparams.arch))

        if self.hparams.arch.startswith("resnet"):
            self.set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft.fc.in_features

            # TODO: Implement advanced pretraining technique from fastai for all models
            if self.hparams.feature_extract == "advanced":
                # Use advanced pretraining technique from fastai - https://docs.fast.ai/vision.learner.html#cnn_learner
                # Remove last two layers
                model_ft = nn.Sequential(*list(model_ft.children())[:-2])
                # append head to resnet body
                model_ft = nn.Sequential(model_ft, self.create_head(num_classes))
            else:
                # Reshape model so last layer has 512 input features and num_classes output features
                model_ft.fc = nn.Linear(num_ftrs, num_classes)

        elif self.hparams.arch.startswith("efficientnet"):
            self.set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft._fc.in_features  # pylint: disable=protected-access
            model_ft._fc = nn.Linear(
                num_ftrs, num_classes
            )  # pylint: disable=protected-access

        elif self.hparams.arch.startswith("alexnet"):
            self.set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

        elif self.hparams.arch.startswith("vgg"):
            self.set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

        elif self.hparams.arch.startswith("squeezenet"):
            self.set_parameter_requires_grad(model_ft)
            model_ft.classifier[1] = nn.Conv2d(
                512, num_classes, kernel_size=(1, 1), stride=(1, 1)
            )
            model_ft.num_classes = num_classes

        elif self.hparams.arch.startswith("densenet"):
            self.set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)

        elif self.hparams.arch.startswith("inception"):
            # Be careful, expects (299,299) sized images and has auxiliary output
            self.set_parameter_requires_grad(model_ft)
            # Handle the auxiliary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)

        else:
            logger.critical("Invalid model name, exiting...")
            exit()

        if self.hparams.relu_to_mish:
            convert_relu_to_mish(model_ft)

        self.classification_model = model_ft

        # Save number of classes in `self.hparams` so it gets saved `on_save_checkpoint`.
        # Save it so that if the model is loaded using `SlideClassifier.load_from_checkpoint`,
        # then the `self.classification_model` can be automatically created.
        self.hparams.num_classes = num_classes

        return model_ft

    def get_input_size(self):
        if self.hparams.arch.startswith("efficientnet"):
            model_type = self.hparams.arch.split("-", 1)[-1]  # get everything after "-"
            if model_type.startswith("b"):
                input_sizes = [224, 240, 260, 300, 380, 456, 528, 600, 672]
                input_size = input_sizes[
                    int(model_type[-1:])
                ]  # select size from array that matches last character of `model_type`
        elif self.hparams.arch.startswith("inception"):
            input_size = 299
        else:
            input_size = 224
        
        self.hparams.input_size = input_size
        return input_size

    def prepare_data(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if self.hparams.use_random_split:
            # Random split dataset
            full_dataset = datasets.ImageFolder(
                self.hparams.data_path,
                transforms.Compose(
                    [
                        transforms.Resize((480, 640)),
                        transforms.CenterCrop(self.hparams.input_size),
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
            traindir = os.path.join(self.hparams.data_path, "train")
            valdir = os.path.join(self.hparams.data_path, "val")
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.Resize((480, 640)),
                        transforms.CenterCrop(self.hparams.input_size),
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
                        transforms.CenterCrop(self.hparams.input_size),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
            classes = train_dataset.classes

        self.hparams.classes = classes
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Initialize the model here because the model needs to know the number of classes,
        # which is found when creating the `torchvision.datasets`. Only initialize if the model
        # has not already been created.
        if self.classification_model is None:
            logger.debug("Model already initialized. Not reinitializing.")
            self.classification_model = self.initialize_model(len(self.hparams.classes))

    def train_dataloader(self):
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.hparams.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,  # see https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/5
        )

        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=self.hparams.workers,
            pin_memory=True,
        )

        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def configure_optimizers(self):
        # Gather the parameters to be optimized/updated
        # If feature_extract is normal then params should only be fully connected layer
        # If feature_extract is advanced then params should be all BatchNorm layers and head (create_head) of network
        # If feature_extract is None then params is model.named_parameters()
        params_to_update = self.parameters()
        logger.debug("Params to learn:\n")
        if self.hparams.feature_extract and self.hparams.feature_extract != "none":
            params_to_update = []
            for name, param in self.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    logger.debug("\t%s", name)
        else:
            params_to_update = self.parameters()
            for name, param in self.named_parameters():
                if param.requires_grad:
                    logger.debug("\t%s", name)

        if self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params_to_update,
                self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )

        elif self.hparams.optimizer == "ranger":
            from ranger.ranger import Ranger

            optimizer = Ranger(
                params_to_update,
                self.hparams.lr,
                k=self.hparams.ranger_k,
                betas=(self.hparams.momentum, self.hparams.optimizer_alpha),
                eps=self.hparams.optimizer_eps,
                weight_decay=self.hparams.weight_decay,
            )

        else:
            optimizer = torch.optim.AdamW(
                params_to_update,
                self.hparams.lr,
                betas=(self.hparams.momentum, self.hparams.optimizer_alpha),
                eps=self.hparams.optimizer_eps,
                weight_decay=self.hparams.weight_decay,
            )

        return optimizer

    def calculate_stats(self, output, target):
        _, preds = torch.max(output.data, 1)

        target_cpu = target.cpu()
        preds_cpu = preds.cpu()
        accuracy = accuracy_score(target_cpu, preds_cpu, normalize=True)

        precision, recall, f_score, support = precision_recall_fscore_support(
            target_cpu, preds_cpu, average="weighted"
        )

        return (
            torch.tensor(accuracy),
            torch.tensor(precision),
            torch.tensor(recall),
            torch.tensor(f_score),
        )

    def training_step(self, batch, batch_idx):
        images, target = batch

        output = self.forward(images)

        loss = self.criterion(output, target)

        accuracy, precision, recall, f_score = self.calculate_stats(output, target)

        # Generate stats for progress bar
        tqdm_dict = {"train_loss": loss, "train_accuracy": accuracy}
        # Generate stats for logs
        log = {
            "train/loss": loss,
            "train/accuracy": accuracy,
            "train/precision": precision,
            "train/recall": recall,
            "train/f_score": f_score,
        }
        output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": log,})
        return output

    def validation_step(self, batch, batch_idx):
        images, target = batch

        output = self.forward(images)

        loss = self.criterion(output, target)

        accuracy, precision, recall, f_score = self.calculate_stats(output, target)

        output = OrderedDict(
            {
                "loss": loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
            }
        )
        return output

    def validation_epoch_end(self, outputs, log_prefix="val"):
        # Get the average loss and accuracy metrics over all evaluation runs
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["accuracy"] for x in outputs]).mean()
        avg_precision = torch.stack([x["precision"] for x in outputs]).mean()
        avg_recall = torch.stack([x["recall"] for x in outputs]).mean()
        avg_f_score = torch.stack([x["f_score"] for x in outputs]).mean()

        # Generate stats for progress bar
        tqdm_dict = {
            log_prefix + "_loss": avg_loss,
            log_prefix + "_accuracy": avg_accuracy,
        }
        # Generate stats for logs
        log = {
            log_prefix + "/loss": avg_loss,
            log_prefix + "/accuracy": avg_accuracy,
            log_prefix + "/precision": avg_precision,
            log_prefix + "/recall": avg_recall,
            log_prefix + "/f_score": avg_f_score,
        }
        result = {
            "progress_bar": tqdm_dict,
            "log": log,
            log_prefix + "_loss": avg_loss,
        }
        return result

    def test_step(self, batch, batch_idx):
        images, target = batch

        output = self.forward(images)

        loss = self.criterion(output, target)

        accuracy, precision, recall, f_score = self.calculate_stats(output, target)

        stats = OrderedDict(
            {
                "loss": loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f_score": f_score,
            }
        )

        prediction = torch.argmax(output, 1)

        result = OrderedDict({"prediction": prediction, "target": target})
        result = {**stats, **result}

        return result

    def test_epoch_end(self, outputs):
        predictions = torch.cat([x["prediction"] for x in outputs], 0).cpu().numpy()
        targets = torch.cat([x["target"] for x in outputs], 0).cpu().numpy()

        plot_confusion_matrix(predictions, targets, self.hparams.classes)
        report = classification_report(
            targets, predictions, target_names=self.hparams.classes
        )
        logging.info(report)

        return self.validation_epoch_end(outputs, log_prefix="test")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])

        parser.add_argument("data_path", metavar="DIR", help="path to dataset")
        parser.add_argument(
            "-a",
            "--arch",
            metavar="ARCH",
            default="resnet34",
            choices=MODEL_NAMES,
            help="model architecture: "
            + " | ".join(MODEL_NAMES)
            + " (default: resnet34)",
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
            "--train_batch_size",
            default=16,
            type=int,
            help="Batch size per GPU/CPU for training.",
        )
        parser.add_argument(
            "--val_batch_size",
            default=16,
            type=int,
            help="Batch size per GPU/CPU for evaluation.",
        )
        parser.add_argument(
            "--test_batch_size",
            default=16,
            type=int,
            help="Batch size per GPU/CPU for testing.",
        )
        parser.add_argument(
            "--momentum",
            default=0.9,
            type=float,
            metavar="M",
            help="momentum. Ranger optimizer suggests 0.95.",
        )
        parser.add_argument(
            "--weight_decay",
            default=1e-2,
            type=float,
            metavar="W",
            help="weight decay (default: 1e-2)",
        )
        parser.add_argument(
            "-k",
            "--ranger_k",
            default=6,
            type=int,
            metavar="K",
            help="Ranger (LookAhead) optimizer k value (default: 6)",
        )
        parser.add_argument(
            "--optimizer_alpha",
            default=0.999,
            type=float,
            metavar="N",
            help="Optimizer alpha parameter (default: 0.999)",
        )
        parser.add_argument(
            "--optimizer_eps",
            default=1e-8,
            type=float,
            metavar="N",
            help="Optimizer eps parameter (default: 1e-8)",
        )
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            action="store_true",
            help="use pre-trained model",
        )
        parser.add_argument(
            "--random_split",
            dest="use_random_split",
            action="store_true",
            help="use random_split to create train and val set instead of train and val folders",
        )
        parser.add_argument(
            "--relu_to_mish",
            action="store_true",
            help="convert any relu activations to mish activations",
        )
        parser.add_argument(
            "--feature_extract",
            choices=["normal", "advanced", "none"],
            default="advanced",
            help="If `False` or `None`, finetune the whole model. When `normal`, only update the reshaped layer params. When `advanced`, use fastai version of feature extracting (add fancy group of layers and only update this group and BatchNorm)",
        )
        parser.add_argument(
            "-o",
            "--optimizer",
            default="adamw",
            help="Optimizer to use (default=AdamW)",
        )

        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        add_help=False, description="PyTorch Slide Classifier Training"
    )

    parser.add_argument(
        "--default_root_dir", type=str, help="Default path for logs and weights",
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="Limits training to a max number number of epochs",
    )
    parser.add_argument(
        "--min_steps",
        default=None,
        type=int,
        help="Limits training to a minimum number number of steps",
    )
    parser.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help="Limits training to a max number number of steps",
    )
    # 1e-4 for whole model # 3e-4 is the best learning rate for Adam, hands down
    parser.add_argument(
        "--lr",
        "--learning_rate",
        default=4e-3,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        default=1,
        type=int,
        help="Check val every n train epochs.",
    )
    parser.add_argument(
        "--gpus",
        default=-1,
        type=int,
        help="Number of GPUs to train on or Which GPUs to train on. (default: -1 (all gpus))",
    )
    parser.add_argument(
        "--overfit_pct",
        default=0.0,
        type=float,
        help="Uses this much data of all datasets (training, validation, test). Useful for quickly debugging or trying to overfit on purpose.",
    )
    parser.add_argument(
        "--train_percent_check",
        default=1.0,
        type=float,
        help="How much of training dataset to check. Useful when debugging or testing something that happens at the end of an epoch.",
    )
    parser.add_argument(
        "--val_percent_check",
        default=1.0,
        type=float,
        help="How much of validation dataset to check. Useful when debugging or testing something that happens at the end of an epoch.",
    )
    parser.add_argument(
        "--test_percent_check",
        default=1.0,
        type=float,
        help="How much of test dataset to check.",
    )
    parser.add_argument(
        "--amp_level",
        type=str,
        default="O1",
        help="The optimization level to use (O1, O2, etc…) for 16-bit GPU precision (using NVIDIA apex under the hood).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        help="Full precision (32), half precision (16). Can be used on CPU, GPU or TPUs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible results. Can negatively impact performace in some cases.",
    )
    parser.add_argument(
        "--profiler",
        action="store_true",
        help="To profile individual steps during training and assist in identifying bottlenecks.",
    )
    parser.add_argument(
        "--progress_bar_refresh_rate",
        default=50,
        type=int,
        help="How often to refresh progress bar (in steps). In notebooks, faster refresh rates (lower number) is known to crash them because of their screen refresh rates, so raise it to 50 or more.",
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        default=5,
        type=int,
        help="Sanity check runs n batches of val before starting the training routine. This catches any bugs in your validation without having to wait for the first validation check.",
    )
    parser.add_argument(
        "--use_logger",
        default="wandb",
        type=str,
        choices=["tensorboard", "wandb"],
        help="Which program to use for logging.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Run the training procedure."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Run the testing procedure."
    )
    parser.add_argument(
        "--load_weights",
        default=False,
        type=str,
        help="Loads the model weights from a given checkpoint",
    )
    parser.add_argument(
        "--load_from_checkpoint",
        default=False,
        type=str,
        help="Loads the model weights and hyperparameters from a given checkpoint.",
    )
    parser.add_argument(
        "-l",
        "--log",
        dest="logLevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: 'Info').",
    )

    parser = SlideClassifier.add_model_specific_args(parser)

    args = parser.parse_args()

    # Setup logging config
    logging.basicConfig(
        format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
        level=logging.getLevelName(args.logLevel),
    )

    if args.seed:
        # Sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
        # More info: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.html#reproducibility
        seed_everything(args.seed)

    if args.load_weights:
        model = SlideClassifier(hparams=args)
        checkpoint = torch.load(
            args.load_weights, map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["state_dict"])

    elif args.load_from_checkpoint:
        model = SlideClassifier.load_from_checkpoint(args.load_from_checkpoint)
        # The model is loaded with self.hparams.data_path set to the directory where the data
        # was located during training. When loading the model, it may be desired to change
        # the data path, which the below line accomplishes.
        if args.data_path:
            model.hparams.data_path = args.data_path

    else:
        model = SlideClassifier(hparams=args)

    if args.use_logger == "wandb":
        wandb_logger = loggers.WandbLogger(
            project="slide-classifier-private", log_model=True
        )
        args.logger = wandb_logger
        models_path = os.path.join(wandb_logger.experiment.dir, "models/")
    else:
        models_path = "models/"

    trainer = Trainer.from_argparse_args(args)

    if args.do_train:
        trainer.fit(model)
    if args.do_test:
        trainer.test(model)
