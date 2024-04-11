#show webcam
import cv2
import numpy as np
import time
import os
import models as md
import torch
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

#count the number of clusters in the density map.
def clustercount(density_map):

    #find the clusters
    density_map = ndimage.label(density_map)[0]
    #find the number of clusters
    maxval = np.max(density_map)

    slices = ndimage.find_objects(density_map)
    
    return maxval,density_map, slices

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("running on", device)
density_model = md.CNNdens()
Magi_1 = md.Magi_1()
Magi_2 = md.Magi_2()
Magi_3 = md.Magi_3()

density_model.load_state_dict(torch.load('mod/density_mod.ckp'))
Magi_1.load_state_dict(torch.load('mod/magi_1.ckp'))
Magi_2.load_state_dict(torch.load('mod/magi_2.ckp'))
Magi_3.load_state_dict(torch.load('mod/magi_3.ckp'))
density_model.eval().to(device)
Magi_1.eval().to(device)
Magi_2.eval().to(device)
Magi_3.eval().to(device)


print("models loaded")

mirror = False

cam = cv2.VideoCapture(5)

while True:

    ret_val, img = cam.read()
    if mirror:
        img = cv2.flip(img, 1)

    img = cv2.resize(img, (1280, 720))

    img = img/255

    density_map = density_model(torch.tensor(img).type(torch.float).unsqueeze(0).permute(0, 3, 1, 2).to('cuda')).cpu()
    
    density_map = density_map.squeeze().detach().numpy()
    
    print(density_map.max())
    
    treshold = -0.1

    #tresold
    density_map[density_map >= treshold] = 1
    density_map[density_map < treshold] = 0

    #gaussian blur
    density_map = cv2.GaussianBlur(density_map, (23, 23), 0)

    number, density_map, slices = clustercount(density_map)
    
    found = False
    idx=0
    magi_ans_letter = []
    for i in slices:
        if i is not None:
            found = True
            rect_size = [i[0].stop-i[0].start,i[1].stop-i[1].start]
            
            if rect_size[0] > 20 and rect_size[1] > 20:
                
                #square the sample and pad it
                sample = img[i[0].start:i[0].stop, i[1].start:i[1].stop]

                sample = cv2.resize(sample, (200, 200))
                cv2.imshow('sample', sample)
                sample = torch.tensor(sample).type(torch.float).unsqueeze(0).permute(0, 3, 1, 2).to('cuda')
                magi_ans = [Magi_1(sample).cpu().detach().numpy(), Magi_2(sample).cpu().detach().numpy(), Magi_3(sample).cpu().detach().numpy()]
                magi_ans_letter.append([])
                for x in magi_ans:
                    
                    output_clean = [0,0,0,0]
                    index = 0
                    for l in x[0]:
                        if l == x.max():
                            output_clean[index] = 1
                            break
                        index += 1

                    match output_clean:
                            case [1,0,0,0]:
                                magi_ans_letter[idx].append("A")
                            case [0,1,0,0]:
                                magi_ans_letter[idx].append("B")
                            case [0,0,1,0]:
                                magi_ans_letter[idx].append("C")
                            case [0,0,0,1]:
                                magi_ans_letter[idx].append("N")
                idx += 1
            else:
                magi_ans_letter.append(["N", "N", "N"])
                idx += 1
    
    print(magi_ans_letter)

    #ONLY USING MAGE
    if found:
        choices = []
        for i in magi_ans_letter:
            choices.append(i[0])

    indx=0
    for i in slices:

        if choices[indx] == "A":
            color = (255, 0, 0)
        elif choices[indx] == "B":
            color = (0, 255, 0)
        elif choices[indx] == "C":
            color = (0, 0, 255)
        elif choices[indx] == "N":
            color = (255, 255, 255)

        if (i is not None) and choices[indx] != "N":
            cv2.rectangle(img, (i[1].start, i[0].start), (i[1].stop, i[0].stop), color, 2)
        indx += 1

    #cv2.imshow('my webcam', density_map.astype(np.float32))

    cv2.imshow('my webcam', img)
    
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()