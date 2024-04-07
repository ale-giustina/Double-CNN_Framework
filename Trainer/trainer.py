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

batch_size = 15
showexample = 0
showdataset = 0

imsize = (720,1280)

transform = T.Compose([T.Resize(imsize,antialias=True)])

transform1 = T.Compose([T.Resize(imsize,antialias=True)])

dataset = m1.CustomImageDataset("Dataset/expansion/ann_expanded", "Dataset/expansion/train_expanded",transform=transform, target_transform=transform1)

validataset = m1.CustomImageDataset("Dataset/annotated", "Dataset/Val",transform=transform, target_transform=transform1)

dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=batch_size)

validataloader = DataLoader(dataset=validataset,shuffle=True,batch_size=batch_size)

print("Datasets created! starting training...")

if showdataset == 1:
    for i in range(len(dataset)):
        image1 = np.transpose(dataset.__getitem__(i)[0], (1, 2, 0))
        label1 = np.transpose(dataset.__getitem__(i)[1], (1, 2, 0))
        fig, axs = plt.subplots(1, 2)  # 1 row, 2 columns

        axs[0].imshow(image1)
        axs[0].set_title('Image 1')

        axs[1].imshow(label1)
        axs[1].set_title('Label 1')

        plt.show()

if showexample != 1:

    image1 = np.transpose(dataset.__getitem__(showexample)[0], (1, 2, 0))
    label1 = np.transpose(dataset.__getitem__(showexample)[1], (1, 2, 0))
    for i in range(1):
        print("Image:",dataset.__getitem__(showexample)[i].shape,"maxval:",np.max(dataset.__getitem__(showexample)[i].detach().numpy()),"minval:",np.min(dataset.__getitem__(showexample)[0].detach().numpy()))


    fig, axs = plt.subplots(1, 2)  # 1 row, 2 columns

    axs[0].imshow(image1)
    axs[0].set_title('Image 1')

    axs[1].imshow(label1)
    axs[1].set_title('Label 1')

    plt.show()

writer = SummaryWriter()
model = m1.CNNet()

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

                validation_loss = m1.calculate_loss_and_accuracy(validataloader, model, criterion, validataset.__getitem__(showexample),model.totalepoch)
                writer.add_scalar('validation_loss', validation_loss, model.steps)

                # Print Loss
                print('Epoch: {}/{} - ({:.2f}%). Validation Loss: {}. '.format(epoch, epochs, epoch*100/epochs , validation_loss))
                
                if validation_loss < model.best_valdiation_loss:
                    model.best_valdiation_loss = validation_loss
                    print('Saving best model')
                    m1.save_model(model)

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

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(35)
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(35)
learning_rate = 0.003
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(20)
learning_rate = 0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(20)