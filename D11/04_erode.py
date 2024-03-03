"""
Corrosion of images
"""
import cv2
import numpy as np

img = cv2.imread('../data/5.png')
cv2.imshow('img', img)

# erosion
kernel = np.ones(shape=(3, 3),
                 dtype=np.uint8)
res = cv2.erode(img,
                kernel,
                iterations=3)
cv2.imshow('res', res)

cv2.waitKey()
cv2.destroyAllWindows()