"""
The outer circular shape of the fitted contour
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
center, radius = cv2.minEnclosingCircle(cnts[0])
center = (int(center[0]), int(center[1]))
radius = int(radius)
print('center:{}, radius:{}'.format(center, radius))
cv2.circle(img, center, radius, (0, 0, 255), 2)
cv2.imshow('res', img)
# A line thickness of -1 is a solidified fill
cv2.circle(gray, center, radius, 255, 2)
cv2.imshow('ggg',gray)

cv2.waitKey()
cv2.destroyAllWindows()