"""
Fitting polygons to contours
"""
import cv2

img = cv2.imread('../data/cloud.png')
cv2.imshow('img', img)

# grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# binarization
t, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('binary', binary)

# Find Outline
cnts, hie = cv2.findContours(binary,
                             cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
# Fitting polygons to contours
adp1 = img.copy()

eps = 0.005 * cv2.arcLength(cnts[0], True)
points = cv2.approxPolyDP(cnts[0], eps, True)
# print(points)
cv2.drawContours(adp1, [points], 0, (0,0,255), 2)
cv2.imshow('adp1', adp1)

# 0.01
adp2 = img.copy()
eps = 0.01 * cv2.arcLength(cnts[0], True)
points = cv2.approxPolyDP(cnts[0], eps, True)
cv2.drawContours(adp2, [points], 0, (0, 0, 255), 2)
cv2.imshow('adp2', adp2)

cv2.waitKey()
cv2.destroyAllWindows()