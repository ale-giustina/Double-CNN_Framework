from bs4 import BeautifulSoup
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


#set to True to save the data to a text file
savetotext = True

xml_filepath = 'Dataset/annotations.xml'

create_images = True

show_images = False

#converts data from the cvat xml format to a list
with open(xml_filepath, 'r') as f:
    data_raw = f.read()

xml_data = BeautifulSoup(data_raw, "xml")

images = xml_data.find_all("image")

data=[]

index = 0

#DATA FORMAT data = [ [image_name, [sixe_x, size_y], [center_x, center_y, radius_x, radius_y, label]], [image_name, [...]] ]

for i in images:
    annotations = i.find_all("ellipse")
    data.append([i["name"], [i["width"],i["height"]], []])
    for x in annotations:
        data[index][2].append([float(x["cx"]), float(x["cy"]), float(x["rx"]), float(x["ry"]), x["label"]])
    index+=1

#save to txt
if savetotext:
    with open('Dataset/annotations.txt', 'w') as f:
        for i in data:
            f.write(i[0] + " " + str(i[1]) + " " + str(i[2]) +"\n")
const=5
if create_images:
    for i in data:
        img = np.zeros((int(i[1][1]), int(i[1][0])), np.uint8)
        for x in i[2]:
            img = cv2.ellipse(img, (int(x[0]), int(x[1])), (int(x[2]/const), int(x[3]/const)), 0, 0, 360, 255, -1)
        if show_images:
            im2=cv2.imread(find(i[0], 'Dataset'))
            fig = plt.figure(frameon=False)
            im2 = plt.imshow(im2)
            im1 = plt.imshow(img, cmap=plt.cm.gray, alpha=0.5)
            plt.show()
        
        #needed if the rotation metadata i squed
        if int(i[1][1])==4000:
            #rotate the image
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imwrite('Dataset/annotated/'+'ann_'+i[0], img)