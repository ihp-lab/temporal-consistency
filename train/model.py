import os
import os.path as ops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models

class resnet18_encoder(nn.Module):
    def __init__(self):
        super(resnet18_encoder, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        resnet18_layers = list(resnet18.children())[:-1] 
        self.resnet18 = nn.Sequential(*resnet18_layers)
        self.fc1 = nn.Linear(512, 256)

    def forward(self, x):
        output = self.resnet18(x)
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        output = F.normalize(output, p=2, dim=1)
        return output

class densenet121_encoder(nn.Module):
    def __init__(self):
        super(densenet121_encoder, self).__init__()
        densenet = models.densenet121(pretrained=True)
        densenet_layers = list(densenet.children())[:-1] 
        self.densenet = nn.Sequential(*densenet_layers)
        self.fc1 = nn.Linear(1024, 256)

    def forward(self, x):
        output = self.densenet(x)
        output = F.relu(output, inplace=True)
        output = F.adaptive_avg_pool2d(output,(1,1))
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        output = F.normalize(output, p=2, dim=1)
        return output

class mobilenet_encoder(nn.Module):
    def __init__(self):
        super(mobilenet_encoder, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        mobilenet_layers = list(mobilenet.children())[:-1] 
        self.mobilenet = nn.Sequential(*mobilenet_layers)
        self.fc1 = nn.Linear(1280, 256)

    def forward(self, x):
        output = self.mobilenet(x)
        output = nn.functional.adaptive_avg_pool2d(output, 1).reshape(output.shape[0], -1)
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        output = F.normalize(output, p=2, dim=1)
        return output


