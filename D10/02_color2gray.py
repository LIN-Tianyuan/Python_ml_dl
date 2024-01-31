"""
Converting color images to grayscale
Cannot convert grayscale images to color images
"""
import cv2

# imread reads color images, defaults to BGR
img = cv2.imread('../data/lena.jpg', 1)  # 0: grayscale
cv2.imshow('img', img)
# Convert color space BGR->GRAY
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

# Actively enters a blocking state and waits for the user to press a key
cv2.waitKey()
cv2.destroyAllWindows()