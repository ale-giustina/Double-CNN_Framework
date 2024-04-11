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

density_model = md.CNNdens()

density_model.load_state_dict(torch.load('mod/density_mod.ckp'))
density_model.eval().to('cuda')

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


    for i in slices:
        if i is not None:
            cv2.rectangle(img, (i[1].start, i[0].start), (i[1].stop, i[0].stop), (0, 255, 0), 2)

    #cv2.imshow('my webcam', density_map.astype(np.float32))
    
    cv2.imshow('my webcam', img)
    
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()