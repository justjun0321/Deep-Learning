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

parser = argparse.ArgumentParser(description='train.py parser')

parser.add_argument('data_dir', type=str, help='Path to dataset')
parser.add_argument('--save_dir', type=str, help='Path to save the checkpoint')
parser.add_argument('--arch', type=str, help='Architecture for model')
parser.add_argument('--learning_rate', type=float, help='Set the learning rate')
parser.add_argument('--hidden_units', type=int, help='Set the hidden units')
parser.add_argument('--epochs', type=int, help='Epochs amount for training')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()

args = vars(parser.parse_args())

data_dir = args['data_dir']
save_path = args['save_dir']
arch = args['arch']
lr = args['learning_rate']
hidden_units = args['hidden_units']
epochs = args['epochs']
gpu = args['gpu']

data_transforms = {'train':transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                [0.229, 0.224, 0.225])]),
                   'test':transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                [0.229, 0.224, 0.225])]),
                   'valid':transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                [0.229, 0.224, 0.225])])}

# TODO: Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train','test','valid']}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders =  {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle = True) for x in ['train','valid','test']}

model = getattr(models, arch)(pretrained = True)
input_size = 25088
output = 102

# Build a feed-forward network
classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_units)),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_units, output)),
                      ('output', nn.LogSoftmax(dim=1))]))

for param in model.parameters():
    param.requires_grad = False

model.classifier = classifier

model.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr)

sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}

def train_model(model, criterion, optimizer, scheduler, num_epochs=epochs, use_gpu = gpu):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if use_gpu:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model, criterion, optimizer, sched, 10, 'cuda')

model.class_to_idx = image_datasets['train'].class_to_idx
torch.save(model, save_path)