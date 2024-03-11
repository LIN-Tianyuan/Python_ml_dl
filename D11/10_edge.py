"""
Edge detection
"""
import cv2

img = cv2.imread('../data/lena.jpg', 0)
cv2.imshow('img', img)
# Sobel
sobel = cv2.Sobel(img,  # imagery
                  cv2.CV_64F, # image depth
                  1,1,  # Derivative order in the horizontal direction in the vertical direction
                  ksize=5) # Size of the operator
cv2.imshow('sobel', sobel)
# Laplacian
lap = cv2.Laplacian(img, cv2.CV_64F)
cv2.imshow('lap', lap)
# Canny
canny = cv2.Canny(img,
                  120,   # low threshold
                  260)  # high threshold
cv2.imshow('canny', canny)
cv2.waitKey()
cv2.destroyAllWindows()
