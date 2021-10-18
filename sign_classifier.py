from numpy import testing
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
import time
import os
import numpy as np
import ntpath
import cv2
import glob
import time

CLASSES = ["circle", "eye","feather", "head", "water"]
preprocess = imagenet_utils.preprocess_input

times = []
print("[INFO] model_loaded classifier...")

def recognize_image(img,model):
    dim = (224, 224)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    image = img_to_array(resized)
    image = np.expand_dims(image, axis=0)
    image = preprocess(image)
    # classify the image
    preds = model.predict(image)
    val = np.argmax(preds, axis=1)
    # printing information
    # print("recognized class {} {}".format(CLASSES[int(val)],preds[0][int(val)]*100))
    for i in range(len(CLASSES)):
        print("class {} {}".format(CLASSES[i],preds[0][i]*100))
    return CLASSES[int(val)],preds[0][int(val)]*100

def loadModel(model_file):
    return load_model(model_file)
