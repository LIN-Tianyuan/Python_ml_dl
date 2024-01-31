"""
Operate on a channel of a color image
"""
import cv2

img = cv2.imread('../data/opencv2.png')
cv2.imshow('img', img)
# Blue channel only (component method for blue)
b = img[:, :, 0]
cv2.imshow('b', b)
# Assign the blue channel in the original image, to 0
img[:, :, 0] = 0
cv2.imshow('b0', img)
# Assign green to 0 on top of blue to 0
img[:, :, 1] = 0
cv2.imshow('b0-g0', img)
cv2.waitKey()
cv2.destroyAllWindows()