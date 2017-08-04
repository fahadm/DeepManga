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
from os import listdir
from os.path import isfile, join
import json
import collections
import itertools
import random
import Data_Loaders.load_data as data_loader
from torch.nn import Parameter

batch_size = 4
use_gpu = torch.cuda.is_available()

def eval_model(model, name, eval_loc = "eval_data"):
    dset_loader, dset_size, dset_classes, dset = data_loader.load_eval_data_from_folder_structure(eval_loc, batch_size=batch_size, use_three_channels=True)
    model.eval()
    running_corrects = 0
    print "dset_size: ", dset_size
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
    print "running corrects: ",running_corrects

    print('Evaluation Result for net: {}: Acc: {:.4f}'.format(name, epoch_acc))
    return epoch_acc

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
#model_list.append((resnet34, "2ClassModels/author_pred_resnet34_best.pth"))
#model_list.append((densenet121, "2ClassModels/author_pred_densenet121_best.pth"))
#model_list.append((resnet50, "2ClassModels/author_pred_resnet50_best.pth"))
model_list.append((densenet169, "2ClassModels/author_pred_densenet169_best.pth"))
#model_list.append((densenet121, "2ClassModels/author_pred_densenet121_0.790909090909.pth"))
#model_list.append((densenet121, "2ClassModels/author_pred_densenet121_0.918181818182.pth"))
#model_list.append((densenet121, "2ClassModels/author_pred_densenet121_0.954545454545.pth"))
#model_list.append((densenet121, "2ClassModels/author_pred_densenet121_0.9.pth"))
#model_list.append((densenet169, "2ClassModels/author_pred_densenet169_0.827272727273.pth"))
#model_list.append((densenet169, "2ClassModels/author_pred_densenet169_0.927272727273.pth"))
#model_list.append((densenet169, "2ClassModels/author_pred_densenet169_0.936363636364.pth"))
#model_list.append((densenet169, "2ClassModels/author_pred_densenet169_0.945454545455.pth"))
#model_list.append((resnet34, "2ClassModels/author_pred_resnet34_0.827272727273.pth"))
#model_list.append((resnet34, "2ClassModels/author_pred_resnet34_0.909090909091.pth"))
#model_list.append((resnet34, "2ClassModels/author_pred_resnet34_0.936363636364.pth"))
#model_list.append((resnet34, "2ClassModels/author_pred_resnet34_0.945454545455.pth"))
#model_list.append((resnet34, "2ClassModels/author_pred_resnet34_0.9.pth"))
#model_list.append((resnet50, "2ClassModels/author_pred_resnet50_0.818181818182.pth"))
#model_list.append((resnet50, "2ClassModels/author_pred_resnet50_0.836363636364.pth"))
#model_list.append((resnet50, "2ClassModels/author_pred_resnet50_0.954545454545.pth"))
#model_list.append((resnet50, "2ClassModels/author_pred_resnet50_0.963636363636.pth"))
#model_list.append((resnet50, "2ClassModels/author_pred_resnet50_0.972727272727.pth"))

best_acc = 0
best_name = ""

for model_func, name in model_list:
    model = model_func(2)
    if use_gpu:
        model.cuda()
    model.load_state_dict(torch.load(name))

    acc = eval_model(model, name)

    if acc > best_acc:
        best_name = name

print "best net: ", best_name