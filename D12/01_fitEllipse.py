"""
Fitting an ellipse to a contour
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
# Fitting ellipse information
params = cv2.fitEllipse(cnts[0])
cv2.ellipse(img, params, color=(0, 0, 255), thickness=2)
cv2.imshow('res', img)
cv2.waitKey()
cv2.destroyAllWindows()