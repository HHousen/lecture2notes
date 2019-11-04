import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def initialize_model(arch, num_classes):
    model = models.__dict__[arch]()
    if arch.startswith("resnet"):
        """ Resnet """
        num_ftrs = model.fc.in_features
        # Reshape model so last layer has 512 input features and num_classes output features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif arch.startswith("alexnet"):
        """ Alexnet """
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif arch.startswith("vgg"):
        """ VGG11_bn """
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif arch.startswith("squeezenet"):
        """ Squeezenet """
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes

    elif arch.startswith("densenet"):
        """ Densenet """
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif arch.startswith("inception"):
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()
        
    return model

# Load saved model
model_best = torch.load("model_best.pth.tar")
class_index = model_best['class_index']
# Load model arch from models
model = initialize_model(model_best['arch'], num_classes=len(model_best['class_index']))
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(model_best['state_dict'])
model.eval()

sm = torch.nn.Softmax()

def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize((480,640)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return my_transforms(image).unsqueeze(0)

def get_prediction(image):
    tensor = transform_image(image)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    probs = sm(outputs).cpu().detach().numpy().tolist()
    predicted_idx = str(y_hat.item())
    class_name = class_index[int(predicted_idx)]
    return class_name, predicted_idx, probs

# print(get_prediction(Image.open("test.png")))