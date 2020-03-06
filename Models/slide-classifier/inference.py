import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from custom_nnmodules import *

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
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
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
        exit()
        
    return model

# Load saved model
MODEL_BEST = torch.load("model_best.pth.tar")
CLASS_INDEX = MODEL_BEST['class_index']
INPUT_SIZE = MODEL_BEST['input_size']
ARCH = MODEL_BEST['arch']
if MODEL_BEST['model']:
    MODEL = MODEL_BEST['model']
    MODEL = MODEL.cuda()
else:
    # Load model arch from models
    MODEL = initialize_model(ARCH, num_classes=len(MODEL_BEST['class_index']))
    MODEL = torch.nn.DataParallel(MODEL).cuda()
MODEL.load_state_dict(MODEL_BEST['state_dict'])
MODEL.eval()

sm = torch.nn.Softmax(dim=1)

def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize((480,640)),
                                        transforms.CenterCrop(INPUT_SIZE),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return my_transforms(image).unsqueeze(0)

def get_prediction(image, percent=False, extract_features=True):
    tensor = transform_image(image)

    if extract_features:
        def copy_data(m, i, o):
            # https://stackoverflow.com/questions/19326004/access-a-function-variable-outside-the-function-without-using-global
            copy_data.extracted_features = torch.clone(o.data).detach().cpu().squeeze()

        # .module needed in both cases below because model wrapped in DataParallel
        if ARCH.startswith("efficientnet"):
            hook = MODEL.module._avg_pooling.register_forward_hook(copy_data)
        else:
            hook = MODEL.module[1][6].register_forward_hook(copy_data)
    else:
        extracted_features = 0

    outputs = MODEL.forward(tensor)

    if extract_features:
        hook.remove()
        extracted_features = copy_data.extracted_features.numpy()

    _, y_hat = outputs.max(1)
    probs = sm(outputs).cpu().detach().numpy().tolist()[0]
    if percent:
        probs = [i*100 for i in probs]
    probs = dict(zip(CLASS_INDEX, probs))
    predicted_idx = int(y_hat.item())
    class_name = CLASS_INDEX[int(predicted_idx)]
    return class_name, predicted_idx, probs, extracted_features

#print(get_prediction(Image.open("test.png")))