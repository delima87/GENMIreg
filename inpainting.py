import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2

R = 2 ** 4

def fill_gaps(im,model):
    im_predict = cv2.resize(im, (im.shape[1] // R * R, im.shape[0] // R * R))
    im_predict = np.reshape(im_predict, (1, im_predict.shape[0], im_predict.shape[1], 1))
    im_predict = im_predict.astype(np.float32) / 255.
    result = model.predict(im_predict)
    im_res = cv2.resize(result[0] * 255., (im.shape[1], im.shape[0]))
    # sharp kernel
    kernel = np.array([[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]])
    im_res_sharp = cv2.filter2D(im_res, -1, kernel)
    return im_res,im_res_sharp

def load_line_inpainting_model(file_model):
    return load_model(file_model)
    
