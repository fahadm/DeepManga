# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader
import time
import copy
import os
import json
import collections
import itertools
import random
import Data_Loaders.load_data as data_loader


# define images and meta-data location
data_dir = 'dev_data'

json_file = 'data.json'

use_gpu = torch.cuda.is_available()

batch_size = 4

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dset_loaders, dset_sizes, dset_classes, dsets = data_loader.load_data_from_folder_structure(data_dir, batch_size=batch_size, use_three_channels=True)
num_classes = 2

print ("Cuda available: ", use_gpu)

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25, save_prefix=""):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            #eval_model(model, save_prefix+'_'+str(best_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
                if save_prefix != "":
                    torch.save(model.state_dict(), './author_pred_'+save_prefix+'_'+str(best_acc)+'.pth')
                else:
                    torch.save(model.state_dict(), './author_pred_net_'+str(best_acc)+'.pth')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return (best_model, 'author_pred_'+save_prefix+'_'+str(best_acc)+'.pth')

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

#define models for this run

def densenet121(num_classes):
    model = torchvision.models.densenet121(pretrained = True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

def densenet169(num_classes):
    model = torchvision.models.densenet169(pretrained = True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

def resnet34(num_classes):
    model = torchvision.models.resnet34(pretrained = True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def resnet50(num_classes):
    model = torchvision.models.resnet50(pretrained = True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

model_list = []
#model_list.append((resnet34, "resnet34"))
#model_list.append((densenet121, "densenet121"))
#model_list.append((resnet50, "resnet50"))
model_list.append((densenet169, "densenet169"))

def train_model_helper(model_ft, criterion, epochs, lr_scheduler, use_gpu, save_prefix = "") :
    if use_gpu:
        model_ft = model_ft.cuda()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    #model_ft = train_model(model_ft, criterion, optimizer_ft, lr_scheduler, num_epochs=num_epochs, save_prefix)
    return train_model(model_ft, criterion, optimizer_ft, lr_scheduler, num_epochs=epochs, save_prefix=save_prefix)

def run_model (model_func, num_classes, lr_scheduler, use_gpu, name_prefix, num_epochs, pretrained_flag=True) :
    model = model_func(num_classes)
    return train_model_helper(model, nn.CrossEntropyLoss(), num_epochs, lr_scheduler, use_gpu, name_prefix)

def eval_model(model, name, eval_loc = "eval_data"):
    dset_loader, dset_size, dset_classes, dset = data_loader.load_eval_data_from_folder_structure(eval_loc, batch_size=batch_size, use_three_channels=True)
    model.eval()
    running_corrects = 0
    for data in dset_loader:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), \
                Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        running_corrects += torch.sum(preds == labels.data)
    epoch_acc = running_corrects / (dset_size * 1.0)
    print('Evaluation Result for net: {}: Acc: {:.4f}'.format(name, epoch_acc))

#run the training
num_epochs = 20

for model_func, name_prefix in model_list:
    best_model, filename = run_model(model_func, num_classes, exp_lr_scheduler, use_gpu, name_prefix, num_epochs, True)
    