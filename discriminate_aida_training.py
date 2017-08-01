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
#
# def flatten (z):
#     return [x for y in z for x in y]
#
# IMG_EXTENSIONS = [
#     '.jpg', '.JPG', '.jpeg', '.JPEG',
#     '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
# ]
#
# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
#
# def get_images_from_folder(dir):
#     return filter(lambda f: is_image_file(f), map (lambda p: os.path.join(dir, p), os.listdir(dir)))
#
# class ImageSet(torch.utils.data.Dataset):
#     def __init__(self, imgs, classes, class_to_idx, transform=None, target_transform=None,
#                  loader=default_loader):
#         if len(imgs) == 0:
#             raise(RuntimeError("Got 0 images: \n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
#         self.imgs = imgs
#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is class_index of the target class.
#         """
#         path, target = self.imgs[index]
#         img = self.loader(path)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target
#
#     def __len__(self):
#         return len(self.imgs)
#
# Manga = collections.namedtuple('Manga', ['title', 'folder_name', 'author'])
# mangas = []
#
# with open(json_file) as data_file:
#     data = json.load(data_file)
#
# for item in data:
#     author_name = item.get("Author")
#     folder_name = item.get("Folder Name")
#     title = item.get("Title")
#     m = Manga(title = title, folder_name = os.path.join(data_dir, folder_name), author = author_name)
#     mangas.append(m)
#
# mangas_by_folder_name = dict()
# for manga in mangas:
#     mangas_by_folder_name[manga.folder_name] = manga
#
# keyfunc = lambda m: m.author
# mangas_by_author = itertools.groupby(sorted(mangas, key=keyfunc), keyfunc)
#
# dirs_by_author = map ( lambda g: ( g[0], map (lambda m: m.folder_name, g[1])), mangas_by_author)
#
# files_by_author = map ( lambda g: ( g[0], flatten( map (get_images_from_folder, g[1]))), dirs_by_author)
#
# class_list = ["Aida", "default"]
# #for k, v in mangas_by_folder_name.iteritems():
# #    class_list.append(v.author)
#
# class_list.sort()
# class_to_idx = {class_list[i]: i for i in range(len(class_list))}
#
# num_classes = len(class_list)
#
# #images = flatten ( map ( lambda g: map ( lambda f: (f, class_to_idx[g[0]]), g[1]), files_by_author) )
# def aida_discrimination(author):
#     if author == "Aida Mayumi":
#         return 0
#     else:
#         return 1
#
# images = flatten ( map ( lambda g: map ( lambda f: (f, aida_discrimination(g[0])), g[1]), files_by_author) )
#
# print (list(images))
#
# num_images = len(images)
# val_set_count = num_images // 10
#
# random.shuffle(images)
# val_set = ImageSet(images[:val_set_count], class_list, class_to_idx, transform = data_transforms["val"])
# train_set = ImageSet(images[val_set_count:], class_list, class_to_idx, transform = data_transforms["train"])
#
# dsets = {"train" : train_set, "val" : val_set}
#
# dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
#                                                shuffle=True, num_workers=4)
#                 for x in ['train', 'val']}
#
# dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
# dset_classes = dsets['train'].classes

dset_loaders, dset_sizes, dset_classes, dsets = data_loader.load_data_from_folder_structure("dev_data", batch_size=batch_size, use_three_channels=True)
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
    return best_model

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
model_list.append((resnet34, "resnet34"))
model_list.append((densenet121, "densenet121"))
model_list.append((resnet50, "resnet50"))
model_list.append((densenet169, "densenet169"))

def train_model_helper(model_ft, criterion, epochs, lr_scheduler, use_gpu, save_prefix = "") :
    if use_gpu:
        model_ft = model_ft.cuda()

    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    #model_ft = train_model(model_ft, criterion, optimizer_ft, lr_scheduler, num_epochs=num_epochs, save_prefix)
    train_model(model_ft, criterion, optimizer_ft, lr_scheduler, num_epochs=epochs, save_prefix=save_prefix)

def run_model (model_func, num_classes, lr_scheduler, use_gpu, name_prefix, num_epochs, pretrained_flag=True) :
    model = model_func(num_classes)
    train_model_helper(model, nn.CrossEntropyLoss(), num_epochs, lr_scheduler, use_gpu, name_prefix)


#run the training
num_epochs = 30

for model_func, name_prefix in model_list:
    run_model(model_func, num_classes, exp_lr_scheduler, use_gpu, name_prefix, num_epochs, True)