"""
Detects the outer edge of the paper using edge detection
"""
import cv2

img = cv2.imread('../data/paper.jpg', 0)
cv2.imshow('img', img)

gau = cv2.GaussianBlur(img, (3, 3), 1)
canny = cv2.Canny(gau,
                  30,
                  100)
cv2.imshow('canny', canny)
cv2.waitKey()
cv2.destroyAllWindows()
