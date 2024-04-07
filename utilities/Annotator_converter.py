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

img_filepath_train = 'Dataset/Train'

ann_filepath = 'Dataset/annotated'

create_images = False

show_images = False

create_duplicates = True

duplicates = 1500

show_duplicates = False

save_duplicates = True

change_size = True
change_rotation = True
change_luminosity = True
change_contrast = True

exp_train = "Dataset/expansion/train_expanded"

exp_ann = "Dataset/expansion/ann_expanded"

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

const=1

if create_images:
    for i in data:

        img = np.zeros((int(i[1][1]), int(i[1][0])), np.uint8)
        img2 = cv2.imread(find(i[0], img_filepath))
        for x in i[2]:
            img = cv2.ellipse(img, (int(x[0]), int(x[1])), (int(x[2]/const), int(x[3]/const)), 0, 0, 360, 255, -1)

        if int(i[1][1])==4000:
            #rotate the image
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if show_images:
            fig = plt.figure(frameon=False)
            im2 = plt.imshow(img2)
            im1 = plt.imshow(img, cmap=plt.cm.gray, alpha=0.5)
            plt.title(str(i[0:2]))
            plt.show()
        

        cv2.imwrite(ann_filepath+'/ann_1'+i[0], img)
        cv2.imwrite(img_filepath_train+'/1'+i[0], img2)


if create_duplicates:
    for i in range(duplicates):
        list_img=os.listdir(img_filepath_train)
        
        rand = np.random.randint(0, len(list_img))
        
        img = cv2.imread(find('ann_'+list_img[rand], ann_filepath))
        img2 = cv2.imread(find(list_img[rand], img_filepath_train))

        #zoom and rotate the image
        if change_size:
            scale = np.random.uniform(0.85, 1.4)
        else:
            scale = 1
        
        if change_rotation:
            angle = np.random.uniform(0, 360)
        else:
            angle = 0
        
        if change_luminosity:
            randoo = np.random.uniform(0.8, 1.4)
            img2 = cv2.convertScaleAbs(img2, alpha=randoo, beta=0)
        if change_contrast:
            randoo = np.random.uniform(-0.5, 0.5)
            img2 = cv2.convertScaleAbs(img2, alpha=1, beta=randoo)

        center = (img.shape[1]//2, img.shape[0]//2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        img2 = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))
        
        if show_duplicates:
            fig = plt.figure(frameon=False)
            im2 = plt.imshow(img2)
            im1 = plt.imshow(img, cmap=plt.cm.gray, alpha=0.5)
            plt.title(data[rand][0:2])
            plt.show()
        
        #random ID
        Id = np.random.randint(0, 1000000)
        
        if save_duplicates:
            cv2.imwrite(exp_ann+f'/ann_{Id}_'+str(i)+data[rand][0], img)
            cv2.imwrite(exp_train+f'/{Id}_'+str(i)+data[rand][0], img2)