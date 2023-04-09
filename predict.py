import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import PIL
from PIL import Image
import argparse
import json

from data_utils import process_image

# Argument parsing for running on Terminal
ap = argparse.ArgumentParser()

ap.add_argument("--input_img", type=str, default="./flowers/test/43/image_02431.jpg", 
                help="image path that we want to predict")
ap.add_argument("--checkpoint", type=str, default="checkpoint.pth",
                help="to get the saved checkpoint file")
ap.add_argument("--top_k", type=int, default=5,
                help="to enter top k prediction classes")
ap.add_argument("--category_names", type=str, default="cat_to_name.json",
                help="to use a mapping of categories to real names")
ap.add_argument("--gpu", type=str, default="gpu",
                help="set GPU mode ON or OFF")


# Assign variable to argument parser
args = ap.parse_args()

input_img_path = args.input_img
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu

# Mapping from category label to category name
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    

# Function to load model's checkpoint
def load_checkpoint(checkpoint_pth):
    """
    Function to load the checkpoint that we saved
    Input: checkpoint.pht
    Return: Model
    """
    checkpoint = torch.load(checkpoint_pth)
    
    arch = checkpoint['arch']
    epoch = checkpoint['epoch']
    gpu = checkpoint['gpu']
    dropout = checkpoint['dropout']
    hidden_layer = checkpoint['hidden_layer']
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
       
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx =  checkpoint['class_to_idx']
    
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model


# Get the model by loading saved checkpoint
model = load_checkpoint(checkpoint)

# Processing the image
processed_image = process_image(input_img_path)


# Function to predict the probs and classes of given image
def predict(processed_image, model, top_k, gpu):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    Return: top probability and classes lists
    '''
    
    # change to evaluation mode
    model.eval()
    
    # process the image
    #img = process_image(image_path)

    # convert from numpy array to TorchTensor 
    img_tensor = torch.from_numpy(processed_image).type(torch.FloatTensor)
    # insert singleton dimension
    img_tensor = img_tensor.unsqueeze_(0)
    
    # Move the model to GPU if GPU is available
    if gpu == "gpu":
        model.to('cuda:0')
    else:
        model.to('cpu')
    
    # turn off gradient and run through the model
    with torch.no_grad():
        if gpu == "gpu":
            img_tensor = img_tensor.to('cuda:0')
        else:
            img_tensor = img_tensor.to('cpu')
        
        output = model.forward(img_tensor)
        
    # calculating probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(top_k)[0]
    index_top = probs.topk(top_k)[1]
    
    # converting from tensor type of probs and classes to lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top)[0]
    
    # getting class_to_idx mapping
    class_to_idx = model.class_to_idx
    # inverting to dictionary for getting idx_to_class mapping too
    idx_to_class = {i: k for k, i in class_to_idx.items()}
    # top class lists
    classes_top_list = [idx_to_class[idx] for idx in index_top_list]
    
    return probs_top_list, classes_top_list


# Print the most likely probability and classes of the image predicted by the model
probs, classes = predict(processed_image, model, top_k, gpu)
print(probs)
print(classes)
flower_name = [cat_to_name[str(i)] for i in classes]
print(f"The predicted flower is most likely to ba '{flower_name[0]}' with a probability of {probs[0]}.")
