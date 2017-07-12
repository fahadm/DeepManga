
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torchvision import datasets, transforms
from torch.autograd import Variable



class convclassifier(nn.Module):
    def __init__(self, input_shape=(1, 256, 256),out_targets = 109):
        super(convclassifier, self).__init__()


        #encoder part
        self.en_conv_1 = nn.Conv2d(1, 100, 5)
        self.en_max_pool_1 =  nn.MaxPool2d(2,return_indices=False)  # b, 16, 5, 5
        self.en_conv_2 = nn.Conv2d(100, 200, 5) # b, 8, 3, 3
        self.en_max_pool_2 = nn.MaxPool2d(2,return_indices=False)  # b, 8, 2, 2

        input_size = self._get_conv_output_size(input_shape)
        self.dense_400 = nn.Linear(input_size,400)
        self.dense_200 = nn.Linear(400, 200)
        self.dense_out = nn.Linear(200, out_targets)

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
    def _get_conv_output_size(self, shape):
        bs = 32
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.en_conv_1(input)
        output_feat = self.en_max_pool_1(output_feat)
        output_feat = self.en_conv_2(output_feat)
        output_feat = self.en_max_pool_2(output_feat)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.en_conv_1(x)
        x = self.en_max_pool_1(x)
        x = self.en_conv_2(x)
        x = self.en_max_pool_2(x)
        x = x.view(x.size(0), -1)

        x = self.dense_400(x)
        x = self.dense_200(x)
        x = self.dense_out(x)


        return F.log_softmax(x)


