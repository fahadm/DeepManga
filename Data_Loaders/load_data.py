import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image

def gs_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:

            return img.convert('L')

def load_data(parent_dir):
    return dset.ImageFolder(root=parent_dir, transform = transforms.ToTensor(), loader=gs_loader)



