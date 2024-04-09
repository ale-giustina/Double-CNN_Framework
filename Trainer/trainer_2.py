import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader
import module1 as m1
from tensorboardX import SummaryWriter
import os
import cv2
import matplotlib.pyplot as plt



device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 60
showexample = 0
showdataset = 0

imsize = (200,200)

dataset = m1.CustomImageDataset_letters("Dataset/Letter_dataset/Ann.csv", "Dataset/Letter_dataset/Train")

validataset = m1.CustomImageDataset_letters("Dataset/Letter_dataset/Ann.csv", "Dataset/Letter_dataset/Val")

dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=batch_size)

validataloader = DataLoader(dataset=validataset,shuffle=True,batch_size=batch_size)

print("Datasets created! starting training...")

if showdataset == 1:
    for i in range(len(dataset)):
        image1 = np.transpose(dataset.__getitem__(i)[0], (1, 2, 0))
        plt.imshow(image1)
        plt.title(dataset.__getitem__(i)[1])
        plt.show()

if showexample != 1:

    image1 = np.transpose(dataset.__getitem__(showexample)[0], (1, 2, 0))

    print("Image:",dataset.__getitem__(showexample)[0].shape,"maxval:",np.max(dataset.__getitem__(showexample)[0].detach().numpy()),"minval:",np.min(dataset.__getitem__(showexample)[0].detach().numpy()))

    plt.imshow(image1)
    plt.title(dataset.__getitem__(showexample)[1])
    plt.show()

    plt.show()


writer = SummaryWriter()
model = m1.CNNet2()

model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)

costval = []

dataiter = iter(dataloader)

steps_before_print = len(dataloader)

lis, labels = next(dataiter)

writer.add_graph(model, lis.to(device))

def train(epochs):

    for epoch in range(epochs):
        
        step = 0
        epoch_loss = 0
        epoch_acc = 0
        
        total_size = len(dataloader)
        
        for i, (images, labels) in enumerate(dataloader):
            
            model.train()

            optimizer.zero_grad()
            
            outputs = model(images.to(device))
            
            loss = criterion(outputs, labels.to(device))
            
            writer.add_scalar('trainning_loss', loss.item(), model.steps)
            
            loss.backward()
            optimizer.step()

            step += 1
            epoch_loss += loss.item()
            print(step, loss.item())

            if step % steps_before_print == 0:
                
                # Calculate Accuracy
                model.eval()

                validation_loss , accuracy = m1.calculate_loss_and_accuracy_letters(validataloader, model, criterion, validataset.__getitem__(showexample),model.totalepoch)
                writer.add_scalar('validation_loss', validation_loss, model.steps)

                # Print Loss
                print('Epoch: {}/{} - ({:.2f}%). Validation Loss: {}.  acc: {}'.format(epoch, epochs, epoch*100/epochs , validation_loss, accuracy))
                
                if accuracy > model.best_acc:
                    model.best_acc = accuracy
                    print('Saving best model')
                    m1.save_model(model, add=accuracy)

                del validation_loss
                
            del loss, outputs, images, labels

        model.epochs += 1
        model.totalepoch += 1
        #print('Epoch({}) avg loss: {} avg acc: {}'.format(epoch, epoch_loss/step, epoch_acc/times_calculated))
        print('Epoch ', epoch)
        #save_model(model, use_ts=True)

#create a log to dump the model characteristics
with open('examples/log.txt', 'w') as f:
    f.write(str(model.charateristics))

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(35)
learning_rate = 0.002
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(40)
