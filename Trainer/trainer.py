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

writer = SummaryWriter()

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 7
showexample = 6

imsize = (720,1280)

transform = T.Compose([T.Resize(imsize,antialias=True)])

transform1 = T.Compose([T.Resize(imsize,antialias=True)])

dataset = m1.CustomImageDataset("Dataset/annotated", "Dataset/Train",transform=transform, target_transform=transform1)

validataset = m1.CustomImageDataset("Dataset/annotated", "Dataset/Val",transform=transform, target_transform=transform1)

dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=batch_size)

validataloader = DataLoader(dataset=validataset,shuffle=True,batch_size=batch_size)

print("Datasets created! starting training...")

if showexample != 0:

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

                validation_loss = m1.calculate_loss_and_accuracy(validataloader, model, criterion, dataset.__getitem__(5),model.totalepoch)
                writer.add_scalar('validation_loss', validation_loss, model.steps)

                # Print Loss
                print('Iteration: {}/{} - ({:.2f}%). Validation Loss: {}. '.format(step, total_size, step*100/total_size , validation_loss))
                
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

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(70)
learning_rate = 0.03
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(40)
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(30)