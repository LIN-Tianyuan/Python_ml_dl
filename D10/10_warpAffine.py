"""
Affine transformations: translation plus rotation
"""
import cv2
import numpy as np


# panning
def translate(img, x, y):
    h, w = img.shape[:2]   # (h,w,c)
    # translation matrix
    M = np.float32([[1, 0, x],
                    [0, 1, y]])
    res = cv2.warpAffine(img,
                         M,
                         (w, h))
    return res


# revolve
def rotate(img, angle, center=None):
    h, w = img.shape[:2]
    if center is None:
        center = (w/2, h/2)

    # rotation matrix
    M = cv2.getRotationMatrix2D(center,    # center of rotation
                                angle,     # angle of rotation
                                1.0)       # zoom ratio
    res = cv2.warpAffine(img,
                         M,
                         (w, h))

    return res


if __name__ == '__main__':
    img = cv2.imread('../data/Linus.png')
    cv2.imshow('img', img)
    # panning
    translated = translate(img, 50, 50)
    cv2.imshow('translated', translated)

    # revolve
    rotated = rotate(img, 45)
    cv2.imshow('rotated', rotated)
    cv2.waitKey()
    cv2.destroyAllWindows()