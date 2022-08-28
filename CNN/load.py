from ast import Attribute
import os
from warnings import catch_warnings
from xml.dom.minidom import AttributeList
import numpy as np
import cv2
from PIL import Image
from warnings import filterwarnings

    
    
# temporal storage for labels and images
data=[]
labels=[]


# Cat 0
# Get the animal directory
cats = os.listdir(os.getcwd() + "/CNN/data/cats")
for x in cats:
    imag=cv2.imread(os.getcwd() + "/CNN/data/cats/" + x)
    try:
        img_from_ar = Image.fromarray(imag,"RGB")
    except AttributeError:
        continue
    
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)



dogs = os.listdir(os.getcwd() + "/CNN/data/dogs")
for x in dogs:
    """
    Loop through all the images in the directory
    1. Convert to arrays
    2. Resize the images
    3. Add image to dataset
    4. Add the label
    """
    imag=cv2.imread(os.getcwd() + "/CNN/data/dogs/" + x)
    try:
        img_from_ar = Image.fromarray(imag,"RGB")
    except AttributeError:
        continue
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)



parrots = os.listdir(os.getcwd() + "/CNN/data/parrots")
for x in parrots:
    
    imag=cv2.imread(os.getcwd() + "/CNN/data/parrots/" + x)
    try:
        img_from_ar = Image.fromarray(imag,"RGB")
    except AttributeError:
        continue
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(2)



monkeys = os.listdir(os.getcwd() + "/CNN/data/monkeys/file-20200409-184019-msyulw")
for x in monkeys:
    """
    Loop through all the images in the directory
    1. Convert to arrays
    2. Resize the images
    3. Add image to dataset
    4. Add the label
    """
    imag=cv2.imread(os.getcwd() + "/CNN/data/monkeys/" + x)
    try:
        img_from_ar = Image.fromarray(imag,"RGB")
    except AttributeError:
        continue
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(3)



bear = os.listdir(os.getcwd() + "/CNN/data/bear")
for x in bear:
    """
    Loop through all the images in the directory
    1. Convert to arrays
    2. Resize the images
    3. Add image to dataset
    4. Add the label
    """
    imag=cv2.imread(os.getcwd() + "/CNN/data/bear/" + x)
    try:
        img_from_ar = Image.fromarray(imag,"RGB")
    except AttributeError:
        continue
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(4)

# load in animals and labels
animals=np.array(data)
labels=np.array(labels)
# save
np.save("animals",animals)
np.save("labels",labels)