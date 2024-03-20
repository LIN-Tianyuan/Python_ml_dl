"""
Image correction
"""
import math

import cv2
import numpy as np

img = cv2.imread('../data/paper.jpg')
cv2.imshow('img', img)
# grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# binarization
# t, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
# cv2.imshow('binary', binary)

# edge detection
# sobel = cv2.Sobel(gray,
#                   cv2.CV_64F,
#                   1, 1,
#                   ksize=5)
# cv2.imshow('sobel', sobel)
# lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
# cv2.imshow('lap', lap)
# blurred
blured = cv2.GaussianBlur(gray,(5, 5), 0)
# closed operation
close = cv2.morphologyEx(blured,
                         cv2.MORPH_CLOSE,
                         (3, 3))
canny = cv2.Canny(close, 30, 120)
cv2.imshow('canny', canny)

# Find Outline
cnts, hie = cv2.findContours(canny,
                             cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
# cv2.drawContours(img, cnts, 2, (0, 0, 255), 2)
# cv2.imshow('cnts', img)
# print(len(cnts))

# Finding the quadrilateral with the largest area: target contours: cv2.contourArea()
docCnt = None
if len(cnts) > 0:
    cnts = sorted(cnts,
                  key=cv2.contourArea,
                  reverse=True)
    for cnt in cnts:
        eps = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) == 4:
            docCnt = approx
            break
# print(docCnt)
img_copy = img.copy()
for peak in docCnt:
    peak = tuple(peak[0])
    cv2.circle(img_copy, peak, 10, (0, 0, 255), 2)
cv2.imshow('img_copy', img_copy)

# Coordinate points before transformation(Top left, bottom left, bottom right, top right.)
src = docCnt.reshape(4, 2).astype('float32')  # (4, 1, 2)

# Find the width and height of the paper
h = int(math.sqrt((src[0][0] - src[1][0]) ** 2 + (src[0][1] - src[1][1]) ** 2))
w = int(math.sqrt((src[0][0] - src[3][0]) ** 2 + (src[0][1] - src[3][1]) ** 2))

# Coordinate points after transformation
dst = np.array([[0, 0],
                [0, h],
                [w, h],
                [w, 0]], dtype='float32')
# perspective transformation matrix
M = cv2.getPerspectiveTransform(src, dst)
# Performing perspective transformations
res = cv2.warpPerspective(img, M, (w, h))
cv2.imshow('res', res)
cv2.waitKey()
cv2.destroyAllWindows()