"""
Expansion of the image
"""
import cv2
import numpy as np

img = cv2.imread('../data/9.png')
cv2.imshow('img', img)

# dilatation
kernel = np.ones(shape=(3, 3),
                 dtype=np.uint8)
res = cv2.dilate(img,
                 kernel,
                 iterations=4)
cv2.imshow('res', res)

cv2.waitKey()
cv2.destroyAllWindows()