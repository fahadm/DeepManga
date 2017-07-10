
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable



class convautoencoder(nn.Module):
    def __init__(self):
        super(convautoencoder, self).__init__()


        #encoder part
        self.en_conv_1 = nn.Conv2d(1, 100, 5)
        self.en_max_pool_1 =  nn.MaxPool2d(2,return_indices=True)  # b, 16, 5, 5
        self.en_conv_2 = nn.Conv2d(100, 200, 5) # b, 8, 3, 3
        self.en_max_pool_2 = nn.MaxPool2d(2,return_indices=True)  # b, 8, 2, 2

        self.de_max_unpool_1 =    nn.MaxUnpool2d(2)
        self.de_conv_1 =     nn.ConvTranspose2d(200,100, 5)
        self.de_max_unpool_2 = nn.MaxUnpool2d(2)
        self.de_conv_2 =  nn.ConvTranspose2d(100, 1, 5)


    def forward(self, x):
        x = self.en_conv_1(x)
        x,indices_1 = self.en_max_pool_1(x)
        x = self.en_conv_2(x)
        x,indices_2 = self.en_max_pool_2(x)

        x = self.de_max_unpool_1(x,indices_2)
        x = self.de_conv_1(x)
        x = self.de_max_unpool_2(x,indices_1)
        x = self.de_conv_2(x)


        return x


