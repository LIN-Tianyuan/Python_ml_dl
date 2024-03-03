"""
Morphological gradients in images
Expansion - Erosion
"""

import cv2
import numpy as np

img = cv2.imread('../data/6.png')
cv2.imshow('img', img)

# morphological gradient
kernel = np.ones(shape=(2, 2),
                 dtype=np.uint8)
res = cv2.morphologyEx(img,
                       cv2.MORPH_GRADIENT,
                       kernel)
cv2.imshow('res', res)
cv2.waitKey()
cv2.destroyAllWindows()