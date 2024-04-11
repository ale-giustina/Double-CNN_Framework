import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode


class CNNdens(nn.Module):
    def __init__(self):
        super(CNNdens, self).__init__()
                
        self.conv1 = nn.Conv2d(3, 10, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=9, stride=1, padding=4)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(20, 15, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.LeakyReLU(inplace=True)
        self.conv5 = nn.Conv2d(15, 1, kernel_size=1, stride=1)
        self.relu5 = nn.LeakyReLU(inplace=True)
        
        self.steps = 0
        self.epochs = 0
        self.totalepoch = 0

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        density_map = self.relu5(x)
                
        if self.training:
            self.steps += 1
                
        return density_map