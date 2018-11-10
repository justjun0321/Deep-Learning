import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
from PIL import Image

parser = argparse.ArgumentParser(description='predict.py parser')

parser.add_argument('path', type=str, help='Path to image')
parser.add_argument('--category_names', type=str, help='Path to model checkpoint')
parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
parser.add_argument('--top_k', type=int, help='Path to model checkpoint')
args = parser.parse_args()

args = vars(parser.parse_args())

image = args['path']
checkpoint = args['checkpoint']
category_names = args['category_names']
top_k = args['top_k']

model = torch.load('filename.pth')

import json

with open(str('aipnd-project/'+category_names+'.json'), 'r') as f:
    cat_to_name = json.load(f)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
        # Resize the width and height
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
        
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img

def predict(image_path, model, top_num=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    # Add batch of size 1 to image
    image_tensor = image_tensor.cuda()
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.cpu()
    top_labs = top_labs.cpu()
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers

top_probs, top_labels, top_flowers = predict(image, model)
for i in range(top_k):
    print('The top',str(i+1),'possible flower is',top_flowers[i],'And the probability is',top_probs[i])