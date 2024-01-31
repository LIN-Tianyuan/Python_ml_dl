"""
Mirroring (flipping)
"""
import cv2

img = cv2.imread('../data/lena.jpg')
cv2.imshow('img', img)

# horizontal mirroring
flip1 = cv2.flip(img, 1)
cv2.imshow('flip1', flip1)

# vertical mirroring
flip0 = cv2.flip(img, 0)
cv2.imshow('flip0', flip0)
cv2.waitKey()
cv2.destroyAllWindows()