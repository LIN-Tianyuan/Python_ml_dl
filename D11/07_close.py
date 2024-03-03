"""
Closed operations on images: first expansion, then erosion
"""
import cv2
import numpy as np

img = cv2.imread('../data/9.png')
cv2.imshow('img', img)

# Closure operation on images to remove internal defects
kernel = np.ones(shape=(1, 3),
                 dtype=np.uint8)
res = cv2.morphologyEx(img,
                       cv2.MORPH_CLOSE,
                       kernel,
                       iterations=15)
cv2.imshow('res', res)
cv2.waitKey()
cv2.destroyAllWindows()