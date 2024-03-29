"""
Cropping of images
Random Crop, Center Crop
"""
import cv2
import numpy as np


def random_crop(img, cw, ch):
    start_x = np.random.randint(0, img.shape[1] - cw)
    start_y = np.random.randint(0, img.shape[0] - ch)
    # slice
    random_res = img[start_y:start_y+ch, start_x:start_x+cw]
    return random_res


# Center Cut
def center_crop(img, cw, ch):
    start_x = int(img.shape[1] / 2) - int(cw / 2)
    start_y = int(img.shape[0] / 2) - int(ch / 2)
    # slice
    random_res = img[start_y:start_y + ch, start_x:start_x + cw]
    return random_res


if __name__ == '__main__':
    img = cv2.imread('../data/banana_1.png')
    cv2.imshow('img', img)
    # Random cropping
    random_res = random_crop(img, 200, 200)
    cv2.imshow('random_res',random_res)
    # Center cropping
    center = center_crop(img, 200, 200)
    cv2.imshow('center', center)
    cv2.waitKey()
    cv2.destroyAllWindows()