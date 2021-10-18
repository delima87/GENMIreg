import numpy as np
import cv2
import random

# function to generate random gaps for input data
def generate_random_gap(imgs, seed=None):
    gap_configs = [
            [10, 200, 2, 3, 0, 1],
            [10, 200, 2, 5, 0, 1],
            [1, 2, 5, 8, 0, 1]
        ]
    bg = np.full(imgs[0].shape, 1., np.float32)
    imgs_with_gaps = []
    masks = []

    if seed is not None:
        np.random.seed(seed)

    for img in imgs:
        img_height, img_width = img.shape[:2]
        mask = np.zeros_like(img, np.float32)
        for gap_config in gap_configs:
            nb_min, nb_max, r_min, r_max, b_min, b_max = gap_config
            _mask = np.zeros_like(img, np.float32)

            for _ in range(np.random.randint(nb_min, nb_max)):
                center = (np.random.randint(img_width), np.random.randint(img_height))
                radius = np.random.randint(r_min, r_max)
                cv2.circle(_mask, center, radius, 1., -1)

                blur_radius = np.random.randint(b_min, b_max) * 2 + 1
                _mask = cv2.blur(_mask, (blur_radius, blur_radius))

                _mask = np.expand_dims(_mask, axis=-1)

            # accumulate masks
            mask = mask + _mask

        mask = np.clip(mask, 0., 1.)

        # composite with mix
        imgs_with_gaps.append(img * (1. - mask) + bg * mask)
        masks.append(mask * (1. - img))

    return np.array(imgs_with_gaps, np.float32), np.array(masks, np.float32)


def rotateImg(img, angle):
    rows = img.shape[0]
    cols = img.shape[1]
    img_center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(img_center, angle, 1)
    print(M)
    return cv2.warpAffine(img, M, (cols, rows), borderValue=(255,255,255))

# function to retrieve segment 
def create_patch_completion(img, gap_size,total_size,is_rect_sign):
    th, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    init_xy = int((total_size - gap_size)/2) #padding to be filled
    offset = init_xy + 8 #8
    base_frame = np.zeros((img.shape[0] + gap_size*2 ,img.shape[1] + gap_size*2),dtype=np.uint8)
    base_frame.fill(255)
    base_frame[offset: offset + img.shape[0], offset: offset + img.shape[1]] = img
    # extract gaps
    if is_rect_sign:
        x = random.randint(offset,(img.shape[1] + offset ) -  gap_size)#int(img.shape[0]/5)
        y = random.randint(offset-5,offset+5)
    else:
        x = random.randint(offset,(img.shape[1] + offset ) -  gap_size)#int(img.shape[0]/5)
        y = random.randint(offset,(img.shape[0] + offset ) -  gap_size)
    crop_img = base_frame[y: y + gap_size,x:x+gap_size]
    full_size = (total_size,total_size)
    completion_img = np.zeros(full_size,dtype=np.uint8)
    completion_img.fill(255)
    completion_img[init_xy:init_xy + gap_size, init_xy:init_xy + gap_size] = crop_img
    randAngle = np.arange(0, 360, step=36)
    ra = random.randint(0,len(randAngle)-1)
    rest_img = base_frame[y - init_xy: (y - init_xy) + (total_size),x - init_xy: (x - init_xy) + (total_size)] 
    completion_img_r = rotateImg(completion_img,randAngle[ra])
    rest_img_r = rotateImg(rest_img,randAngle[ra])
    return completion_img_r,rest_img_r 