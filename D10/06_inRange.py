"""
Extracts the specified color
"""
import cv2
import numpy as np

img = cv2.imread('../data/opencv2.png')
cv2.imshow('img', img)

# BGR --> HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Extract Blue, H:120
min_val = np.array([110, 50, 50])
max_val = np.array([130, 255, 255])

mask = cv2.inRange(hsv, min_val, max_val)
cv2.imshow('mask', mask)
# The original image and the original image do bitwise with
res = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('res', res)


cv2.waitKey()
cv2.destroyAllWindows()