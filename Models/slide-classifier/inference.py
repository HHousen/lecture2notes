import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from custom_nnmodules import *

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
if model_best['model']:
    model = model_best['model']
    model = model.cuda()
else:
    # Load model arch from models
    model = initialize_model(model_best['arch'], num_classes=len(model_best['class_index']))
    model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(model_best['state_dict'])
model.eval()

sm = torch.nn.Softmax(dim=1)

def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize((480,640)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return my_transforms(image).unsqueeze(0)

def attach_hook(model):
    my_embedding = torch.zeros(512)
    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # Attach that function to our selected layer
    h = model[1][6].register_forward_hook(copy_data)
    # Run the model on our transformed image
    model(t_img)
    # Detach our copy function from the layer
    h.remove()
    # Return the feature vector
    return my_embedding

def get_prediction(image, percent=False):
    tensor = transform_image(image)

    extracted_features = torch.zeros(1, 512)
    def copy_data(m, i, o):
        extracted_features.copy_(o.data)
    hook = model.module[1][6].register_forward_hook(copy_data)

    outputs = model.forward(tensor)

    hook.remove()
    extracted_features = extracted_features.numpy()[0, :]

    _, y_hat = outputs.max(1)
    probs = sm(outputs).cpu().detach().numpy().tolist()[0]
    if percent:
        probs = [i*100 for i in probs]
    probs = dict(zip(class_index, probs))
    predicted_idx = int(y_hat.item())
    class_name = class_index[int(predicted_idx)]
    return class_name, predicted_idx, probs, extracted_features

#print(get_prediction(Image.open("test.png")))