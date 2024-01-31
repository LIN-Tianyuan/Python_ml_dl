"""
Binarization and inverse binarization
"""
import cv2

img = cv2.imread('../data/lena.jpg')
# grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

# binarization
t, binary = cv2.threshold(gray,
                          100,  # thresholds
                          255,  # Greater than threshold goes to 255
                          cv2.THRESH_BINARY)    # binarization
cv2.imshow('binary', binary)

cv2.waitKey()
cv2.destroyAllWindows()