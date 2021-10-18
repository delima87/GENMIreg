import cv2
import numpy as np
from matplotlib import pyplot as plt
import random 
import regUtils as regutils
import inpainting as inp
from completion import predict_completion
import sign_classifier as sc
def create_random_puzzles(imgFile):
    img = cv2.imread(imgFile,0)
    img1,img1C = regutils.create_patch_completion(img,64,112,True)
    img2,img2C = regutils.create_patch_completion(img,64,112,True)
    img1_gaps = get_generated_randoms(img1)
    img2_gaps = get_generated_randoms(img2)
    return img1_gaps,img1C,img2_gaps,img2C

def get_generated_randoms(img):
    y = np.expand_dims(img, -1) / 255
    s = random.randint(0,100)
    x, m = regutils.generate_random_gap([y], s)
    return  x[0] * 255


          

if __name__ == "__main__":
    # inputs
    inFile = "pairwise_puzzle\eye_im2d.jpg"
    inpaintingModelFile = "ModelsOffline\model1.h5"
    signClassifierModelFile = "ModelsOffline\signs_recog_remote_sensing_30.model"
    inpaintingModel= inp.load_line_inpainting_model(inpaintingModelFile)
    signClassifierModel = sc.loadModel(signClassifierModelFile)
    # create puzzles
    img1, img1C, img2, img2C = create_random_puzzles(inFile)
    # inpainting
    img1_inp,a = inp.fill_gaps(img1,inpaintingModel)
    img2_inp,b = inp.fill_gaps(img2,inpaintingModel)
    # classify signs
    img1_inp_c = cv2.cvtColor(img1_inp,cv2.COLOR_GRAY2RGB)
    img2_inp_c = cv2.cvtColor(img2_inp,cv2.COLOR_GRAY2RGB)
    sign_classA, sign_scoreA = sc.recognize_image(img1_inp_c,signClassifierModel)
    sign_classB, sign_scoreB = sc.recognize_image(img2_inp_c,signClassifierModel)
    print("classified CLASSES",sign_classA,sign_classB)
    # save images in file for completion
    cv2.imwrite("temp/testA/imA.jpg",img1_inp)
    cv2.imwrite("temp/testA/imB.jpg",img2_inp)
    cv2.imwrite("temp/testB/imA.jpg",img1C)
    cv2.imwrite("temp/testB/imB.jpg",img2C)
    # completion
    checkpoints = "./checkpoints"
    name_model =  sign_classA + "_remote_sensing" 
    data = "./temp"
    predicted_imgs = predict_completion(data,name_model,checkpoints)
    # visualization
    img1 = np.reshape(img1, (112,112))
    img2 = np.reshape(img2, (112,112))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Random Segments of Ancient Inscriptions')
    ax1.imshow(predicted_imgs[1],cmap='gray')
    ax2.imshow(predicted_imgs[3],cmap='gray')
    plt.show()  



