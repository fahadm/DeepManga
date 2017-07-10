from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import Models.CAE as CAE
import Data_Loaders.load_data as load_data
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from torchvision import datasets, transforms
from torch.autograd import Variable


batch_size = 8
lr = 3E-4
momentum = 0.9

epochs = 30

cuda = False

log_interval = 20

# #torch.manual_seed(seed)
# if cuda:
#     torch.cuda.manual_seed(seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
   load_data.load_data("../Data/Manga109_processed/images"),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    load_data.load_data("../Data/Manga109_processed/images"),
    batch_size=batch_size, shuffle=True, **kwargs)


model = CAE.convautoencoder()
if cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 256, 256)
    return x
criterion = nn.MSELoss()

def train(epoch):
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


for epoch in range(1, epochs + 1):
    train(epoch)

