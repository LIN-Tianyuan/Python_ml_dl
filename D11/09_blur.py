"""
Blurring of images: reducing pixel-to-pixel differences
"""
import cv2
import numpy as np

img = cv2.imread('../data/salt.jpg')
cv2.imshow('img', img)

# Mean value filter
blured = cv2.blur(img, (5, 5))
cv2.imshow('blured', blured)

# Gaussian filter
gaussian = cv2.GaussianBlur(img,
                            (5, 5),
                            3)
cv2.imshow('gaussian', gaussian)

# Median filter
median = cv2.medianBlur(img, 5)
cv2.imshow('median', median)

# Specify the convolution kernel yourself and perform the convolution
filter_w = np.ones(shape=(5, 5),
                   dtype='float32') / 25.0

res = cv2.filter2D(img, # Original image
                   -1,  # Depth of the image
                   filter_w)    # Convolution kernel
cv2.imshow('res', res)
cv2.waitKey()
cv2.destroyAllWindows()
