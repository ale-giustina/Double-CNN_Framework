import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import math
import time
import os
import matplotlib.pyplot as plt
import pandas as pd

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
        label = read_image(label_path,ImageReadMode.GRAY).type(torch.float)
        #convert image and label to float
        image = image/255
        label = (label-255/2)/(255/2)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label #convert to float from unit8

class CustomImageDataset_letters(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.imlist = os.listdir(img_dir)

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imlist[idx])
        image = read_image(img_path).type(torch.float)

        label = self.imlist[idx][0]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        image = image/255

        match label:
                case "A":
                    label = [1,0,0,0]
                case "B":
                    label = [0,1,0,0]
                case "C":
                    label = [0,0,1,0]
                case "N":
                    label = [0,0,0,1]

        label = torch.tensor(label).type(torch.float)

        return image, label


class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
                
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
        self.best_valdiation_loss = math.inf
        self.totalepoch = 0
        self.charateristics = "self.conv1 = nn.Conv2d(3, 10, kernel_size=7, stride=1, padding=3)\n\
        self.relu1 = nn.LeakyReLU(inplace=True)\n\
        self.conv2 = nn.Conv2d(10, 20, kernel_size=9, stride=1, padding=4)\n\
        self.relu2 = nn.LeakyReLU(inplace=True)\n\
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2)\n\
        self.relu3 = nn.LeakyReLU(inplace=True)\n\
        self.conv4 = nn.Conv2d(20, 15, kernel_size=3, stride=1, padding=1)\n\
        self.relu4 = nn.LeakyReLU(inplace=True)\n\
        self.conv5 = nn.Conv2d(15, 1, kernel_size=1, stride=1)\n\
        self.relu5 = nn.LeakyReLU(inplace=True)\n\
        radius_constant = 1    "

        
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



class CNNet2(nn.Module):
    def __init__(self):
        super(CNNet2, self).__init__()
                
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(10, 1, kernel_size=1, stride=1)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.view = nn.Flatten()
        self.linear = nn.Linear(200*200, 60)
        self.relu4 = nn.LeakyReLU(inplace=True)
        self.linear2 = nn.Linear(60, 4)
        self.relu5 = nn.LeakyReLU(inplace=True)
        
        self.steps = 0
        self.epochs = 0
        self.best_valdiation_loss = math.inf
        self.totalepoch = 0
        self.charateristics = "self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)\n\
        self.relu1 = nn.LeakyReLU(inplace=True)\n\
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)\n\
        self.relu2 = nn.LeakyReLU(inplace=True)\n\
        self.conv3 = nn.Conv2d(10, 1, kernel_size=1, stride=1)\n\
        self.relu3 = nn.LeakyReLU(inplace=True)\n\
        self.view = nn.Flatten()\n\
        self.linear = nn.Linear(200*200, 60)\n\
        self.relu4 = nn.LeakyReLU(inplace=True)\n\
        self.linear2 = nn.Linear(60, 4)\n\
        self.relu5 = nn.LeakyReLU(inplace=True)"

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.view(x)
        x = self.linear(x)
        x = self.relu4(x)
        x = self.linear2(x)
        x = self.relu5(x)
                
        if self.training:
            self.steps += 1
                
        return x

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
        # Total number of labels
        steps += 1

        del outputs, loss

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

        

    return total_loss/steps


def save_model(model, use_ts=False,curr_folder='./mdl'):
    if not os.path.exists(curr_folder):
        os.makedirs(curr_folder) 
    if use_ts:
        time_stamp = time.strftime("%d_%b_%Y_%Hh%Mm", time.gmtime())
        torch.save(model, curr_folder + '/{}.ckp'.format(time_stamp))        #model.state_dict()
    else:
        torch.save(model, curr_folder + '/{}.ckp'.format('best_model'))        #model.state_dict()

def calculate_loss_and_accuracy_letters(validation_loader, model, criterion,example_image,epoch):
    
    total = 0
    steps = 0
    total_loss = 0
    sz = len(validation_loader)
    correct = 0
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

        #calculate accuracy
        for i in range(len(outputs)):
            output = outputs[i].cpu().detach().tolist()
            label = labels[i].cpu().detach().tolist()
            output_clean = [0,0,0,0]
            
            index = 0
            for i in output:
                if i == np.array(output).max():
                    output_clean[index] = 1
                    break
                index += 1
            
            if output_clean == label:
                correct += 1
            total += 1
            # Total number of labels
        steps += 1

        del outputs, loss

    #save an image of the output, the input and the label
    output_example = model(example_image[0].cuda()).cpu().detach().numpy()
    input_example = example_image[0]
    label_example = example_image[1]

    match label_example.tolist():
            case [1,0,0,0]:
                label_example = "A"
            case [0,1,0,0]:
                label_example = "B"
            case [0,0,1,0]:
                label_example = "C"
            case [0,0,0,1]:
                label_example = "N"

    output_clean = [0,0,0,0]
    index = 0
    for i in output_example[0]:
        if i == output_example.max():
            output_clean[index] = 1
            break
        index += 1

    match output_clean:
            case [1,0,0,0]:
                output_clean = "A"
            case [0,1,0,0]:
                output_clean = "B"
            case [0,0,1,0]:
                output_clean = "C"
            case [0,0,0,1]:
                output_clean = "N"

    fig , ax = plt.subplots(2, height_ratios=[1,1.5])

    fig.tight_layout(pad=3.0)

    ax[0].imshow(np.transpose(input_example,(1,2,0)))

    plt.title(str(label_example)+ '  ->  ' + str(output_clean) + "      acc: " + str(correct/total))

    ax[1].bar(['A','B','C','N'],output_example[0])

    #plt.show()
    plt.savefig('examples/{}.png'.format(epoch), dpi=800)
    plt.close(fig)


    return total_loss/steps , correct/total


def save_model(model, use_ts=False,curr_folder='./mdl'):
    if not os.path.exists(curr_folder):
        os.makedirs(curr_folder) 
    if use_ts:
        time_stamp = time.strftime("%d_%b_%Y_%Hh%Mm", time.gmtime())
        torch.save(model, curr_folder + '/{}.ckp'.format(time_stamp))        #model.state_dict()
    else:
        torch.save(model, curr_folder + '/{}.ckp'.format('best_model'))        #model.state_dict()