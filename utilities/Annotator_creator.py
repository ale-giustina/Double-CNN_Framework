#create the images for the second CNN
from bs4 import BeautifulSoup
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import PIL.ExifTags

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


#set to True to save the data to a text file
savetotext = False

xml_filepath = 'Dataset/annotations.xml'

img_filepath = 'Dataset'

img_filepath_train = 'Dataset/Letter_dataset/Train'

csv_filepath = 'Dataset/Letter_dataset/Ann.csv'

show_images = False

duplicate = 4

#
#converts data from the cvat xml format to a list
#

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

#open the csv file
label_file = open(csv_filepath, 'w')

#creates the images

for i in data:
    for x in i[2]:
        
        img = cv2.imread(find(i[0], img_filepath))
        
        center = [int(x[0]), int(x[1])]

        adding = np.random.randint(13, 35)

        size = int((x[2]+x[3])/2)+adding

        img = img[center[1]-size:center[1]+size, center[0]-size:center[0]+size]

        if img.shape[0] != 0 and img.shape[1] != 0:

            #resize the image

            size = 200

            img = cv2.resize(img, (size, size))

            label = [0,0,0,0]

            match x[4]:
                case "A":
                    label = [1,0,0,0]
                case "B":
                    label = [0,1,0,0]
                case "C":
                    label = [0,0,1,0]

            if show_images:
                fig = plt.figure(frameon=False)
                im1 = plt.imshow(img)
                plt.title(str(x[4]))
                plt.show()
            
            if duplicate != 0:
                for _ in range(duplicate):
                    #random ID
                    Id = np.random.randint(0, 1000000)

                    randoo = np.random.uniform(-0.5, 0.5)
                    img2 = cv2.convertScaleAbs(img, alpha=1, beta=randoo)

                    randoo = np.random.uniform(0.8, 1.4)
                    img3 = cv2.convertScaleAbs(img2, alpha=randoo, beta=0)

                    angle = np.random.uniform(0, 360)
                    M = cv2.getRotationMatrix2D((size/2,size/2), angle, 1)
                    img4 = cv2.warpAffine(img3, M, (size,size))

                    cv2.imwrite(img_filepath_train+'/'+str(str(x[4])+"_"+str(Id)+i[0]), img4)

                    label_file.write(str(str(x[4])+"_"+str(Id)+i[0]) + "," + x[4] + "\n")

            #random ID
            Id = np.random.randint(0, 1000000)

            cv2.imwrite(img_filepath_train+'/'+str(x[4])+"_"+str(Id)+i[0], img)

            label_file.write(str(str(x[4])+"_"+str(Id)+i[0]) + "," + x[4] + "\n")

