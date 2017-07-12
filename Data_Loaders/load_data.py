import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
import json
from pprint import pprint


def gs_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:

            return img.convert('L')

def load_data(parent_dir):
    return dset.ImageFolder(root=parent_dir, transform = transforms.ToTensor(), loader=gs_loader)


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





