# Flower-Classifier
This is the second project I've submitted in AI Programming with Python Nano Degree Program. This project use [the flower dataset of 102 category](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) and train with transfer learning of pretrained arcitecture. The project goal is to recognize different species of flower with the trained model. The workflow steps can be seen in [the notebook](https://github.com/ThuraTunScibotics/Flower-Classifier/blob/main/Flower_Classifier_workflow.ipynb). Then, the steps in the notebook are separated into specific python scripts and run with python argument parser.

### To **train** the model
Train the model on the training set of the data by defining the device for training on GPU with desired pre-trained architecure, and other hyperparameters like learning rate, hidden units and epochs.
```
python train.py --data_directory /path/to/train
```
  * Options;
    * Set the directory to save the checkpoint: ```python train.py --data_dir /path/to/train --save_dir /path/to/save```
    * Choose the architecture: ```python train.py --data_dir --arch "alexnet"```
    * Set hyperparameters: ```python train.py --data_dir /path/to/train --learning_rate 0.01 --hidden_units 512 --epochs 20```
    * Use GPU for training: ```python train.py --data_dir /path/to/train --gpu```
    
### For **Prection** using the trained model
 By using the trained checkpoint/model single image of flower is predicted, and return flower name and class probability. The flower image for prediction can be obtained from the test folder of the dataset.
```
python predict.py --input_img /path/to/test/ --checkpoint /checkpoint.pth
```
* Options;
  * Return top K most likely classes; ```python predict.py --input_img /path/to/test/ --checkpoint /checkpoint.pth --top_k 3```
  * Use a mapping of categories to real names; ```python predict.py --input_img /path/to/test/ --checkpoint /checkpoint.pth --category_names /cat_to_name.json```
  * Use GPU for inference; ```python predict.py --input_img /path/to/test/ --checkpoint /checkpoint.pth --gpu```
 
    
