"""
Defect detection in the dial area
"""
import cv2
import numpy as np

img = cv2.imread('../data/CPU3.png')
cv2.imshow('img', img)
# grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
# binarization
t, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
cv2.imshow('binary', binary)
# Find the outline of the dial area
cnts, hie = cv2.findContours(binary,
                             cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
# print(len(cnts))
# cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
# cv2.imshow('cnts', img)

# Generate a solid black image exactly the size of the binary
mask = np.zeros_like(binary)
cv2.imshow('mask', mask)

# Draw the outline of the dial on the mask using a solidified fill.
img_fill = cv2.drawContours(mask, cnts, -1, 255, -1)
cv2.imshow('img_fill', img_fill)

# image subtraction img_fill - mask get defects
img_sub = cv2.subtract(img_fill, binary)
cv2.imshow('img_sub', img_sub)

# Closure operation on defects, shrinking discrete points
close = cv2.morphologyEx(img_sub,
                         cv2.MORPH_CLOSE,
                         (3, 3),
                         iterations=2)
cv2.imshow('close', close)

# Finding the outline of a blemish
cnts, hie = cv2.findContours(close,
                             cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
# print(len(cnts))
# Sort the defects by area and take the defect with the largest area.
if len(cnts) > 0:
    cnts = sorted(cnts,
                  key = cv2.contourArea,
                  reverse=True)
    # Calculate the area of the defect with the largest area
    area = cv2.contourArea(cnts[0])
    print(area)
    if area > 10:   # industry standard
        # Fitting the Minimum Outer Circle of a Maximum Defect
        center, radius = cv2.minEnclosingCircle(cnts[0])
        center = int(center[0]), int(center[1])
        radius = int(radius)
        # Drawing out imperfections in the original image
        cv2.circle(img, center, radius, (0, 0, 255), 2)
        print('Flawed, Flawed area: ', area)
        cv2.imshow('res', img)

cv2.waitKey()
cv2.destroyAllWindows()