import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from datetime import datetime
import PIL
from PIL import Image
import argparse

from network_archi import network_classifier
from data_utils import load_data


# Argument parsing for running on command line
ap = argparse.ArgumentParser()

ap.add_argument("--data_dir", type=str, default="./flowers/", 
                help="data directory for getting data")
ap.add_argument("--save_dir", type=str, default="./checkpoint.pth", 
                help="directory to save checkpoint")
ap.add_argument("--arch", type=str, default="vgg16", 
                help="choosing pretrained architecture")
ap.add_argument("--learning_rate", type=float, default=0.001, 
                help="to set the learning rate")
ap.add_argument("--hidden_units", type=int, default=1024, 
                help="hidden layer of the classifier model")
ap.add_argument("--gpu", type=str, default="gpu", 
                help="set gpu mode ON or OFF")
ap.add_argument("--epochs", type=int, default=5, 
                help="to set numbers of epoch")
ap.add_argument("--dropout", type=float, default=0.2, 
                help="to set dropout for preventing overfitting")

# Assign variables to argument parser
args = ap.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
lr = args.learning_rate
hidden_layer = args.hidden_units
gpu = args.gpu
epochs = args.epochs
dropout = args.dropout

# Function to train the model
def train_model(model, criterion, optimizer, trainloader, validationloader, epochs, gpu):
    """
    Train the model
    Return: losses and accuracy of training and validation
    """
    train_losses, valid_losses = [], []
    
    for e in range(epochs):
        t0 = datetime.now()
        total_train_loss = 0
        
        for images, labels in trainloader:
            if gpu == "gpu":
                images, labels = images.to("cuda:0"), labels.to("cuda:0")
            else:
                images, labels = images.to("cpu"), labels.to("cpu")
        
            optimizer.zero_grad()
        
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
        
            optimizer.step()
        
            total_train_loss += loss.item()
        
        else:
            total_valid_loss = 0
            accuracy = 0
            model.eval()
        
            # turn off gradient since we're not updating parameters during validation pass
            with torch.no_grad():
                for images, labels in validloader:
                    if gpu == "gpu":
                        images, labels = images.to("cuda:0"), labels.to("cuda:0")
                    else:
                        images, labels = images.to("cpu"), labels.to("cpu")
                
                    log_ps = model(images)
                    batch_loss = criterion(log_ps, labels)
                
                    total_valid_loss += batch_loss.item()
                
                    # calculate the accuracy
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
        
            # Get the mean loss to compare between train and validation loss
            train_loss = total_train_loss / len(trainloader.dataset)
            valid_loss = total_valid_loss / len(validloader.dataset)
        
            # Save the losses
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
        
            dt = datetime.now() - t0
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Training Loss: {:.3f}...".format(train_loss),
                  "Validation Loss: {:.3f}...".format(valid_loss),
                  "Validation Accuracy: {:.3f}...".format(accuracy/len(validloader)),
                  "Duration: {}".format(dt))
            

# Get the necessary data-sets
train_data, trainloader, validloader, testloader = load_data(data_dir)

# Get the model architecture and others parameters
model, criterion, optimizer = network_classifier(arch, hidden_layer, dropout, lr)

# Train the model
print("-----TRAINING STARTS-----")
train_model(model, criterion, optimizer, trainloader, validloader, epochs, gpu)


# Function for doing validation on the test dataset
def test_model(model, testloader, gpu):
    
    test_accuracy = 0
    model.eval()
    
    # turn off gradient since we're not updating parameters for testing
    with torch.no_grad():
        for images, labels in testloader:
            
            if gpu == "gpu":
                images, labels = images.to("cuda:0"), labels.to("cuda:0")
            else:
                images, labels = images.to("cpu"), labels.to("cpu")
    
            # Forward pass
            log_ps = model(images)
    
            # Get prediction probability
            ps = torch.exp(log_ps)
        
            top_p, top_class = ps.topk(1, dim=1)
        
            equality = top_class == labels.view(*top_class.shape)
        
            test_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
        
    print("Test Accuracy: {:.3f}".format(test_accuracy/len(testloader)))
    
    
# Test the network
print("-----START VALIDATION ON THE TEST SET-----")
test_model(model, testloader, gpu)

# Saving the model checkpoint
print("-----SAVING THE MODEL CHECKPOINT-----")
model.class_to_idx = train_data.class_to_idx
checkpoint = {'arch': arch,
              'hidden_layer': hidden_layer,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'epoch': epochs,
              'optimizer_state_dict': optimizer.state_dict(),
              'gpu': gpu,
              'dropout': dropout}

torch.save(checkpoint, save_dir)
print("---FINISHED---")