"""
Open arithmetic of images: first corrosion, then expansion
"""
import cv2
import numpy as np

img = cv2.imread('../data/5.png')
cv2.imshow('img', img)

# open operator (computing)
kernel = np.ones(shape=(3, 3),
                 dtype=np.uint8)
res = cv2.morphologyEx(img,
                       cv2.MORPH_OPEN,
                       kernel,
                       iterations=3)
cv2.imshow('res', res)
cv2.waitKey()
cv2.destroyAllWindows()