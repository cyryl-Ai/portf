import os
import cv2
from PIL import Image
import numpy as np

import tensorflow as tf

data=[]
labels=[]

parrots = os.listdir(os.getcwd() + "/CNN/data/parrots")
for x in parrots:
    
    imag=cv2.imread(os.getcwd() + "/CNN/data/parrots/parrots.jpg")
    try:
        img_from_ar = Image.fromarray(imag,"RGB")
    except AttributeError:
        continue
    resized_image = img_from_ar.resize((50, 50))
        
    
    test_image =np.expand_dims(resized_image,axis=0)

    model = tf.keras.models.load_model(os.getcwd() + '/model.h5')

    result= model.predict(test_image)
    print(result)
    print("Result is: ", result[0][0])
    print("Prediction: " + str(np.argmax(result)))