import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Dictionay for choosing pre-trained network architecture
pretrained = {"vgg16":25088,
              "densenet121":1024,
              "alexnet":9216}


# Network architecture function
def network_classifier(arch="vgg16", hidden_layer=1024, dropout=0.2, lr=0.003):
    """
    Pre-trained Network Model that have been trained on very large dataset included with feature detector and 
    classifier, and its classifier need to replace based on our specific problem.
    
    Input: vgg16 is default, and other models: densenet121, alexnet, resnet can be used
    Return: model
    """
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("Sorry! try with vgg16, densenet121, or alexnet")    
    
    # Freeze the models parameters as we don't want to backprop them-turn off gradient
    for param in model.parameters():
        model.requires_grad = False
    
    # Create specific classifier  
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(pretrained[arch], hidden_layer)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_layer, 512)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(dropout)),
                          ('fc3', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    model.to(device)
    
    return model, criterion, optimizer
    