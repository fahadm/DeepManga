import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import torch
from pprint import pprint

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def gs_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:

            return img.convert('L')

def load_data(parent_dir,transform = transforms.ToTensor()):
    return dset.ImageFolder(root=parent_dir, transform = transform, loader=gs_loader)


def load_target_mapping(filename = "mapping.json"):
    with open(filename) as data_file:
        data = json.load(data_file)
        #collapse authors
        author = {}
        authors = []
        invertedMap = {}
        for object in data:
            a_name = object["Author"]
            f_name = object["Folder Name"]
            vol_name = object["Volume in dataset"]
            print f_name
            invertedMap[f_name] = a_name
            if vol_name is not None:
                invertedMap[f_name+"_vol{0:02d}".format(vol_name)] = a_name
            if not author.has_key(a_name) :
                author[a_name] = []
                authors.append(a_name)
            author[a_name].append(f_name)
            invertedMap[f_name] = a_name

        return  author, invertedMap, authors

def load_data_from_folder_structure(parent_dir, transform = data_transforms, batch_size = 4, use_three_channels = False):
    data_dir = parent_dir
    if use_three_channels:
        dsets = {x: dset.ImageFolder(os.path.join(data_dir, x), transform[x])
                 for x in ['train', 'val']}
    else:
        dsets = {x: dset.ImageFolder(os.path.join(data_dir, x), transform[x], loader=gs_loader)
                 for x in ['train', 'val']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    return dset_loaders, dset_sizes, dset_classes, dsets