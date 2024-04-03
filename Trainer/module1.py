import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import math
import time
import os
from scipy import ndimage

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
        image = read_image(img_path)
        label_path = os.path.join(self.labels_dir, "ann_"+self.imlist[idx])
        label = read_image(label_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

class net(nn.Module):
	def __init__(self,input_size,output_size):
		super(net,self).__init__()
		
		self.relu = nn.LeakyReLU()
		self.l1 = nn.Linear(input_size,126)
		self.l2 = nn.Linear(126,126)
		self.l3 = nn.Linear(126,50)
		self.l4 = nn.Linear(50,output_size)
		self.sftmx = nn.Softmax(dim=1)

		self.steps = 0
		self.epochs = 0
		self.best_valdiation_loss = math.inf

	def forward(self,x):
		
		output = self.l1(x) 
		output = self.relu(output)
		output = self.l2(output)
		output = self.relu(output)
		output = self.l3(output)
		output = self.relu(output)
		output = self.l4(output)
		print(output.shape)
		if self.training: output = self.sftmx(output)

		if self.training:
			self.steps += 1
		
		return output

class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()
		
		# Encoder
		self.encoder_conv1 = nn.Conv2d(3, 64, 3, padding=1)
		self.encoder_conv2 = nn.Conv2d(64, 128, 3, padding=1)
		self.encoder_pool = nn.MaxPool2d(2, 2)
		self.steps = 0
		self.epochs = 0
		self.best_valdiation_loss = math.inf
		# Decoder
		self.decoder_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.decoder_conv1 = nn.Conv2d(256, 64, 3, padding=1)
		self.decoder_conv2 = nn.Conv2d(64, 64, 3, padding=1)
		self.decoder_conv3 = nn.Conv2d(64, 1, 1)  # Output density map with 1 channel
		
	def forward(self, x):
		# Encoder
		x1 = nn.functional.relu(self.encoder_conv1(x))
		x2 = nn.functional.relu(self.encoder_conv2(x1))
		x3 = self.encoder_pool(x2)
		
		# Decoder
		x4 = self.decoder_upsample(x3)
		x4 = torch.cat((x4, x2), dim=1)
		x5 = nn.functional.relu(self.decoder_conv1(x4))
		x5 = nn.functional.relu(self.decoder_conv2(x5))
		x6 = self.decoder_conv3(x5)
		if self.training:
			self.steps += 1
		return x6

class DensityMapModel(nn.Module):
	def __init__(self):
		super(DensityMapModel, self).__init__()
				
		# Define the layers of your model
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
		self.relu1 = nn.LeakyReLU(inplace=True)
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
		self.relu2 = nn.LeakyReLU(inplace=True)
		self.conv3 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
		self.relu3 = nn.LeakyReLU(inplace=True)
		self.steps = 0
		self.epochs = 0
		self.best_valdiation_loss = math.inf
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.conv3(x)
		density_map = self.relu3(x)
				
		if self.training:
			self.steps += 1
				
		return torch.squeeze(density_map)


def clustercount2(density_map, treshold, mxlen=17):
	density_map = (density_map >= treshold) * density_map
	density_map = ndimage.measurements.label(density_map)[0]
	maxval = np.max(density_map)
	return maxval


def calculate_loss_and_accuracy(validation_loader, model, criterion, stop_at = 1200, print_every=99999):
	correct = 0
	total = 0
	steps = 0
	total_loss = 0
	sz = len(validation_loader)
	delta=[]
	for images, labels in validation_loader:
	
		if total%print_every == 0 and total > 0:
			accuracy = 100 * correct / total
			print(accuracy)
		
		if total >= stop_at:
			break;
		if torch.cuda.is_available():
			images = images.cuda()
			labels = labels.cuda()

		# Forward pass only to get logits/output
		outputs = model(images)
		
		#Get Loss for validation data
		loss = criterion(outputs, labels)
		total_loss += loss.item()

		# Get predictions from the maximum value
		predicted=clustercount2(outputs.to('cpu').detach(),1)
		labels=clustercount2(labels.to('cpu'),1)

		delta.append(abs(predicted-labels))

		# Total number of labels
		total += 1
		steps += 1

		correct += (predicted == labels).sum().item()

		del outputs, loss, predicted

	accuracy = 100 * correct / total
	deltaval=np.mean(np.array(delta))
	return total_loss/steps, accuracy,deltaval


def save_model(model, use_ts=False,curr_folder='./mdl'):
	if not os.path.exists(curr_folder):
		os.makedirs(curr_folder) 
	if use_ts:
		time_stamp = time.strftime("%d_%b_%Y_%Hh%Mm", time.gmtime())
		torch.save(model, curr_folder + '/{}.ckp'.format(time_stamp))		#model.state_dict()
	else:
		torch.save(model, curr_folder + '/{}.ckp'.format('best_model'))		#model.state_dict()