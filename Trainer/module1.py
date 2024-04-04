import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import math
import time
import os
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, transform=None, target_transform=None):
        self.labels_dir = annotations_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.imlist = os.listdir(img_dir)

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imlist[idx])
        image = read_image(img_path).type(torch.float)
        label_path = os.path.join(self.labels_dir, "ann_"+self.imlist[idx])
        label = read_image(label_path).type(torch.float)
        #convert image and label to float
        image = image/255
        label = (label-255/2)/(255/2)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label #convert to float from unit8

#TODO: NORMALIZE THE DATAAAAAAAAAAAAAAAAA

class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
                
        self.conv1 = nn.Conv2d(3, 10, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=9, stride=1, padding=4)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(20, 15, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(15, 1, kernel_size=1, stride=1)
        self.relu4 = nn.LeakyReLU(inplace=True)
        
        self.steps = 0
        self.epochs = 0
        self.best_valdiation_loss = math.inf
        self.totalepoch = 0

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        density_map = self.relu4(x)
                
        if self.training:
            self.steps += 1
                
        return density_map




def calculate_loss_and_accuracy(validation_loader, model, criterion,example_image,epoch):
    
    total = 0
    steps = 0
    total_loss = 0
    sz = len(validation_loader)
    delta=[]
    
    for images, labels in validation_loader:
    
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        # Forward pass only to get logits/output
        outputs = model(images)
        
        #Get Loss for validation data
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        #save an image of the output, the input and the label
        output_example = model(example_image[0].cuda()).cpu().detach().numpy()
        input_example = example_image[0]
        label_example = example_image[1]
        #create a matplotlib plot
        fig, ax = plt.subplots(2,2)
        
        ax[0][0].imshow(np.transpose(input_example,(1,2,0)))
        
        ax1=ax[1][0].imshow(np.transpose(output_example,(1,2,0)))
        fig.colorbar(ax1, ax=ax[1][0])
        
        ax2=ax[0][1].imshow(np.transpose(output_example,(1,2,0)) > 0)
        fig.colorbar(ax2, ax=ax[0][1])
        
        ax[1][1].imshow(np.transpose(label_example,(1,2,0)))
        
        
        #plt.show()
        plt.savefig('examples/{}.png'.format(epoch), dpi=800)
        plt.close(fig)

        # Total number of labels
        steps += 1

        del outputs, loss

    return total_loss/steps


def save_model(model, use_ts=False,curr_folder='./mdl'):
    if not os.path.exists(curr_folder):
        os.makedirs(curr_folder) 
    if use_ts:
        time_stamp = time.strftime("%d_%b_%Y_%Hh%Mm", time.gmtime())
        torch.save(model, curr_folder + '/{}.ckp'.format(time_stamp))        #model.state_dict()
    else:
        torch.save(model, curr_folder + '/{}.ckp'.format('best_model'))        #model.state_dict()