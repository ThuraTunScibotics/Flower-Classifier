import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import PIL
from PIL import Image

# Function for loading data
def load_data(data_dir):
    """
    Load the train, validation, test data and processing the data.
    
    Input: data_dir- directory of flower datasets
    Return: trainloader, validationloader, testloader
    """
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return train_data, trainloader, validloader, testloader


# Function to process image that will be used for predicting
def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model.
    Input: image
    Return: processed image
    """
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    # Resize the image with shortest size
    # keeping aspect ratio
    if img.size[0] > img.size[1]:
        shortest_height = 256
        height_percent = (shortest_height/float(img.size[1]))
        aspect_width = int(float(img.size[0]) * float(height_percent))
        img = img.resize((aspect_width, shortest_height), PIL.Image.NEAREST)
    else:
        shortest_width = 256
        width_percent = (shortest_width/float(img.size[0]))
        aspect_height = int(float(img.size[1]) * float(width_percent))
        img = img.resize((shortest_width, aspect_height), PIL.Image.NEAREST)
    
    # crop out the center of the image
    left = (img.size[0] - 224)/2
    top = (img.size[1] - 224)/2
    right = (img.size[0] + 224)/2
    bottom = (img.size[1] + 224)/2
    img = img.crop((left, top, right, bottom))
    
    # Normalize the image
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean)/std
    
    # Order the dimension
    img = img.transpose((2, 0, 1))
    
    return img