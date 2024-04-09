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

duplicate = 7

ex_per_image = 5

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

size_final = 200

for i in data:
    
    #create empty examples

    img = cv2.imread(find(i[0], img_filepath))
    img2 = np.zeros((int(i[1][1]), int(i[1][0]),3), np.uint8)
    for x in i[2]:
        img2 = cv2.ellipse(img2, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), 0, 0, 360, (255,255,255), -1)

    img = np.maximum(img, img2)

    for x in range(ex_per_image):
        edge_x = np.random.randint(200, img.shape[1]-size_final)
        edge_y = np.random.randint(200, img.shape[0]-size_final)
        save = img[edge_y:edge_y+size_final, edge_x:edge_x+size_final]
        
        Id = np.random.randint(0, 1000000)
        cv2.imwrite(img_filepath_train+'/'+"N_"+str(Id)+"_"+str(x)+i[0], save)
        label_file.write("N_"+str(Id)+"_"+str(x)+i[0] + "," + "N" + "\n")

    for x in i[2]:
        
        img = cv2.imread(find(i[0], img_filepath))
        
        adding = np.random.randint(10, 46)

        size = int((x[2]+x[3])/2)+adding

        skew = np.random.randint(-10, 20)

        #add padding to the image
        img = cv2.copyMakeBorder(img, abs(skew)//2, abs(skew)//2, abs(skew)//2, abs(skew)//2, cv2.BORDER_CONSTANT, value=[0,0,0])

        center = [int(x[0])+skew, int(x[1])+skew]

        img = img[center[1]-size:center[1]+size, center[0]-size:center[0]+size]

        if img.shape[0] != 0 and img.shape[1] != 0:

            #resize the image

            img = cv2.resize(img, (size_final, size_final))

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
                    M = cv2.getRotationMatrix2D((size_final/2,size_final/2), angle, 1)
                    img4 = cv2.warpAffine(img3, M, (size_final,size_final))

                    cv2.imwrite(img_filepath_train+'/'+str(str(x[4])+"_"+str(Id)+i[0]), img4)

                    label_file.write(str(str(x[4])+"_"+str(Id)+i[0]) + "," + x[4] + "\n")

            #random ID
            Id = np.random.randint(0, 1000000)

            cv2.imwrite(img_filepath_train+'/'+str(x[4])+"_"+str(Id)+i[0], img)

            label_file.write(str(str(x[4])+"_"+str(Id)+i[0]) + "," + x[4] + "\n")

