import cv2
import numpy as np
from matplotlib import pyplot as plt
import random 
import regUtils as regutils
import inpainting as inp
import sign_classifier as sc



def create_random_puzzles(imgFile):
    img = cv2.imread(imgFile,0)
    img1,img1C = regutils.create_patch_completion(img,64,112,True)
    img2,img2C = regutils.create_patch_completion(img,64,112,True)
    img1_gaps = get_generated_randoms(img1)
    img2_gaps = get_generated_randoms(img2)
    return img1_gaps,img1,img1C,img2_gaps,img2,img2C

def get_generated_randoms(img):
    y = np.expand_dims(img, -1) / 255
    s = random.randint(0,100)
    x, m = regutils.generate_random_gap([y], s)
    return  x[0] * 255




