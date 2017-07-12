from __future__ import print_function
import argparse

import sys
import torch
import torch.nn as nn
from datetime import time
import numpy as np
import Models.CAE as CAE
import Models.CAE_Classify as Classify

import Data_Loaders.load_data as data_loader
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from torchvision import datasets, transforms
from torch.autograd import Variable


batch_size = 4
lr = 3E-4
momentum = 0.9

epochs = 30

cuda = False

log_interval = 20

# #torch.manual_seed(seed)
# if cuda:
#     torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


def load_data():
    global train_loader
    global  test_loader
    global mapping, inveretedMap, authors
    mapping, inveretedMap, authors = data_loader.load_target_mapping();

    train_loader = torch.utils.data.DataLoader(
    data_loader.load_data("../Data/Manga109_processed/images"),
    batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
    data_loader.load_data("/home/fahadm/DLCV_Final_Project/Data/Manga109_processed/Test"),
    batch_size=batch_size, shuffle=True, **kwargs)



def map_target_to_author(target, loader ):
    arr = []
    for x in target.numpy():
        arr.append( authors.index( inveretedMap[loader.dataset.classes[x]]) )

    return torch.from_numpy(np.asarray(arr))

def load_model():

    pretrained_dict =  torch.load("./Outputs/conv_autoencoder.pth", map_location=lambda storage, loc: storage)
    model.load_my_state_dict(pretrained_dict)
    # model_dict = model.state_dict()
    #
    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # # 3. load the new state dict
    # model.load_state_dict(pretrained_dict)
def load_pretrained():
    pretrained_dict =  torch.load("/media/fahadm/Media/Trained Models/100/conv_autoencoder.pth", map_location=lambda storage, loc: storage)
    model.load_my_state_dict(pretrained_dict)

def select_model(mode = "classify"):
    global  model
    global optimizer
    global  criterion
    if mode == "cae":
        model = CAE.convautoencoder()
        criterion = nn.MSELoss()

    elif mode == "classify":
        model = Classify.convclassifier(out_targets=len(mapping))
        criterion = nn.NLLLoss()
    if cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)



def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 256, 256)
    return x

def train_CAE(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), data.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, epochs, loss.data[0]))
    if epoch % 1 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './image_{}.png'.format(epoch))
        torch.save(model.state_dict(), './conv_autoencoder.pth')


def train_classifier(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = map_target_to_author(target,train_loader)
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, epochs, loss.data[0]))
    if epoch % 1 == 0:
        torch.save(model.state_dict(), './conv_autoencoder.pth')






def test_classifier(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        old_target = target.numpy()
        target = map_target_to_author(target,train_loader)
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        print ("Input " + target)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(argv):
    start_time = time()

    load_data()
    select_model()
    # load_model()
    load_pretrained()
    test_classifier(0)
    print ("Training without acquisition")
    # for epoch in range(1, epochs + 1):
    #     train_classifier(epoch)
    #     test_classifier(epoch)

    print("--- %s seconds ---" % (time() - start_time))


if __name__ == '__main__':
    main(sys.argv[1:])
