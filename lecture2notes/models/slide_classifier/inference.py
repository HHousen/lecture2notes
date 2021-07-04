import sys

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from .custom_nnmodules import *  # noqa: F401,F403
from .slide_classifier_pytorch import SlideClassifier


def initialize_model(arch, num_classes):
    model = models.__dict__[arch]()
    if arch.startswith("resnet"):
        num_ftrs = model.fc.in_features
        # Reshape model so last layer has 512 input features and num_classes output features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif arch.startswith("alexnet"):
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif arch.startswith("vgg"):
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif arch.startswith("squeezenet"):
        model.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model.num_classes = num_classes

    elif arch.startswith("densenet"):
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif arch.startswith("inception"):
        # Be careful, expects (299,299) sized images and has auxiliary output
        # Handle the auxiliary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        sys.exit()

    return model


def load_model_deprecated(model_path="model_best.pth.tar"):
    """Load saved model trained using old script (".pth.tar" file extension is old format)."""
    model_best = torch.load(model_path)
    class_index = model_best["class_index"]
    input_size = model_best["input_size"]
    arch = model_best["arch"]
    if model_best["model"]:
        model = model_best["model"]
        model = model.cuda()
    else:
        # Load model arch from models
        model = initialize_model(arch, num_classes=len(model_best["class_index"]))
        model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(model_best["state_dict"])
    model.eval()

    model_info = {
        "class_index": class_index,
        "input_size": input_size,
        "arch": arch,
    }

    return model, model_info


def load_model(model_path="model_best.ckpt"):
    """Load saved model from `model_path`."""
    model = SlideClassifier.load_from_checkpoint(model_path)
    model.eval()
    return model


sm = torch.nn.Softmax(dim=1)


def transform_image(image, input_size=224):
    my_transforms = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return my_transforms(image).unsqueeze(0)


def get_prediction(model, image, percent=False, extract_features=True):
    tensor = transform_image(image, model.hparams.input_size)

    if extract_features:

        def copy_data(m, i, o):
            # https://stackoverflow.com/questions/19326004/access-a-function-variable-outside-the-function-without-using-global
            copy_data.extracted_features = torch.clone(o.data).detach().cpu().squeeze()

        # ".classification_model" needed since actual model is stored in that attribute
        # (model.forward simply calls model.classification_model.forward).
        if model.hparams.arch.startswith("efficientnet"):
            hook = model.classification_model._avg_pooling.register_forward_hook(
                copy_data
            )
        else:
            hook = model.classification_model[2].register_forward_hook(copy_data)
    else:
        extracted_features = 0

    outputs = model.forward(tensor)

    if extract_features:
        hook.remove()
        extracted_features = copy_data.extracted_features.numpy()

    _, y_hat = outputs.max(1)
    probs = sm(outputs).cpu().detach().numpy().tolist()[0]
    if percent:
        probs = [i * 100 for i in probs]
    probs = dict(zip(model.hparams.classes, probs))
    predicted_idx = int(y_hat.item())
    class_name = model.hparams.classes[int(predicted_idx)]
    return class_name, predicted_idx, probs, extracted_features


# from PIL import Image
# model = SlideClassifier.load_from_checkpoint("model.ckpt")
# model.eval()
# print(model)
# print(get_prediction(model, Image.open("test.jpg"), None, extract_features=True))
