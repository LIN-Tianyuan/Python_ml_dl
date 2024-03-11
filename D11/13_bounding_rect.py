"""
The outer rectangle of the fitted contour
"""
import cv2
import numpy as np

img = cv2.imread('../data/cloud.png')
cv2.imshow('img', img)
# grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
# binarization
t, binary = cv2.threshold(gray,
                          127,255,
                          cv2.THRESH_BINARY)
cv2.imshow('binary', binary)
# Find Outline
cnts, hie = cv2.findContours(binary,
                             cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)

# print(len(cnts))
# Generate parameters for fitting the contour based on the contour coordinates
x, y, w, h = cv2.boundingRect(cnts[0])
points = np.array([[[x, y]],  # top left
                  [[x, y+h]],  # bottom left
                  [[x+w, y+h]],  # bottom right
                  [[x+w, y]]])  # top right
res = cv2.drawContours(img,
                       [points],
                       0,
                       (0,0,255),
                       2)
cv2.imshow('res', res)
cv2.imshow('qqq', img)


cv2.waitKey()
cv2.destroyAllWindows()