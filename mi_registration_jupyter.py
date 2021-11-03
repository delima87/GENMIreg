import numpy as np
from numpy.lib.type_check import imag 
import cv2
from regUtils import rotateImg
import random
from matplotlib import pyplot as plt
import os

def mutual_information(im1,im2):
    hgram, x_edges, y_edges = np.histogram2d(
    im1.ravel(),
    im2.ravel(),
    bins=20)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    MI = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

class MIStructure:
    def __init__(self,_mi,_tx,_ty,_angle):
        self.mi = _mi
        self.tx = _tx
        self.ty = _ty
        self.angle = _angle

def getOptimalTransformation(query,target):
    mi_values =  []
    mi_class = []
    for tx in range(-query.shape[1],query.shape[1],8):
        print("row")
        for ty in range(-query.shape[0],query.shape[0],8):
            for angle in range (0,360,36):
                query_t = apply_transformation(query,angle,tx,ty)
                mi = mutual_information(query_t,target)
                cur_mi = MIStructure(mi,tx,ty,angle)
                mi_class.append(cur_mi)
                mi_values.append(mi)
    mi_np = np.asarray(mi_values)
    s_id = np.argsort(mi_np)
    # print(mi_values[s_id[-2]])
    # saving best imgs
    best_imgs = []
    mi_heights = []
    for i in range(-1,-10,-1):
        # print("mi",mi_class[s_id[i]].mi)
        best_img = stitch_imgs_canvas(query,mi_class[s_id[i]].angle, mi_class[s_id[i]].tx,mi_class[s_id[i]].ty,target)
        best_imgs.append(best_img)
        mi_heights.append(round(mi_class[s_id[i]].mi,5))
        # file =  outdir + "/best_img_" + str(i) + "_mi_" + str(round(mi_class[s_id[i]].mi,5)) + ".jpg"
    return best_imgs            
                
def apply_transformation(img, angle, tx, ty):
    rows = img.shape[0]
    cols = img.shape[1]
    img_center = (cols / 2, rows / 2)
    img_center = (cols / 2, rows / 2)
    R = cv2.getRotationMatrix2D(img_center, angle, 1)
    M = np.float32([[R[0][0], R[0][1], tx], [R[1][0], R[1][1],ty]])
    return cv2.warpAffine(img, M, (cols, rows), borderValue=(255,255,255))

def get_grid(x, y, homogenous=False):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1]))) if homogenous else coords

def stitch_imgs_canvas(image, _angle, tx,ty,target):
    # Grid to represent image coordinate
    height, width = image.shape[:2]
    img_center = (width / 2, height / 2)
    Rot = cv2.getRotationMatrix2D(img_center, _angle, 1)
    M = np.float32([[Rot[0][0], Rot[0][1], tx], [Rot[1][0], Rot[1][1],ty]])
    coords = get_grid(width, height, True)
    x_ori, y_ori = coords[0], coords[1] 
    # Apply transformation
    warp_coords = M@coords.astype(np.int)
    xcoord2, ycoord2 = warp_coords[0, :], warp_coords[1, :]
    # # Map the pixel RGB data to new location in another array
    ypix_int = y_ori.astype(int)
    xpix_int = x_ori.astype(int)
    canvas = np.zeros_like(image)
    canvas = np.zeros((height*3, width*3),dtype=np.uint8)
    canvas.fill(255)
    for i in range(len(xcoord2)):
        pty = int(ycoord2[i] + height)
        ptx = int(xcoord2[i] + width)
        canvas[pty,ptx] = image[ypix_int[i],xpix_int[i]]
    canvas[height:height + target.shape[1],width:width + target.shape[0]] = target
    return canvas


def sitch_imgs(img,tx,ty,angle,img_b):
    print(tx,ty)
    rows = img.shape[0]
    cols = img.shape[1]
    canvas = np.zeros((rows*3, cols*4),dtype=np.uint8)
    canvas.fill(255)
    canvas[rows + ty: rows*2 + ty, cols + tx :cols*2 + tx] = img
    canvas = rotateImg(canvas,angle)
    canvas[rows: rows*2, cols:cols*2] = img_b
    print(rows + ty,rows*2 + ty)
    print(cols + tx,cols*2 + tx)
    return canvas





# if __name__ == "__main__":
#     file_name = "input_data\pairwise_puzzle\scale_eyea.png"
#     input_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
#     # generate images
#     leftAimg, leftBimg, rightAimg, rightBimg = random_square_gaps(input_img,64,112)
#     leftBimg_30 = rotateImg(leftBimg,60)
#     rightBimg_30 = rotateImg(rightBimg,30)
#     getOptimalTransformation(leftBimg_30,rightBimg_30)
#     # cv2.imshow("left",leftBimg_30)
#     # cv2.imshow("right",rightBimg)

    
    