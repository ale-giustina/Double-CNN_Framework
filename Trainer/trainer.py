import torch
import torch.nn as nn
import torch.nn.functional as F
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

batch_size = 14
steps_before_print = 50
step_size = 2
showexample = 15

#TODO: create datasets

dataset = m1.CustomImageDataset("Dataset/annotated", "Dataset/Train")

validataset = m1.CustomImageDataset("Dataset/annotated", "Dataset/Val")

dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=batch_size)

validataloader = DataLoader(dataset=validataset,shuffle=True,batch_size=batch_size)

print("Datasets created! starting training...")

if showexample != 0:

    image1 = np.transpose(dataset.__getitem__(showexample)[0], (1, 2, 0))
    label1 = np.transpose(dataset.__getitem__(showexample)[1], (1, 2, 0))

    print(dataset.__getitem__(showexample)[0].shape)
    print(dataset.__getitem__(showexample)[1].shape)

    fig, axs = plt.subplots(1, 2)  # 1 row, 2 columns

    axs[0].imshow(image1)
    axs[0].set_title('Image 1')

    axs[1].imshow(label1)
    axs[1].set_title('Label 1')

    plt.show()


#TODO: create model
model = m1.CNNet()

model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)

costval = []

dataiter = iter(dataloader)

lis, labels = next(dataiter)

writer.add_graph(model, lis.to(device))

def train(epochs):
    
    for epoch in range(epochs):
        
        step = 0
        epoch_loss = 0
        epoch_acc = 0
        times_calculated = 0
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


            if step % steps_before_print == 0:
                
                # Calculate Accuracy
                model.eval()
                #TODO: calculate accuracy
                validation_loss, accuracy, deltaval = m1.calculate_loss_and_accuracy(validataloader, model, criterion, stop_at = 1200)
                writer.add_scalar('validation_loss', validation_loss, model.steps)
                writer.add_scalar('accuracy', accuracy, model.steps)
                writer.add_scalar('deltaval', deltaval, model.steps)
                epoch_acc += accuracy
                times_calculated += 1
                
                # Print Loss
                print('Iteration: {}/{} - ({:.2f}%). Loss: {}. Accuracy: {}, deltaavg:{}'.format(step, total_size, step*100/total_size , loss.item(), accuracy,deltaval))
                
                if abs(deltaval) < model.best_valdiation_loss:
                    model.best_valdiation_loss = deltaval
                    print('Saving best model')
                    m1.save_model(model)
                del validation_loss
                
            del loss, outputs, images, labels

        model.epochs += 1

        #print('Epoch({}) avg loss: {} avg acc: {}'.format(epoch, epoch_loss/step, epoch_acc/times_calculated))
        print('Epoch ', epoch)
        #save_model(model, use_ts=True)

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(50)
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(20)
learning_rate = 0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
train(20)