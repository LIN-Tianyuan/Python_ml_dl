"""
Addition of images
"""
import cv2

lena = cv2.imread('../data/lena.jpg', 0)
lily = cv2.imread('../data/lily_square.png', 0)

cv2.imshow('lena', lena)
cv2.imshow('lily', lily)

# additive
add_res = cv2.add(lena, lily)
cv2.imshow('add_res', add_res)
# Summing by weight
add_w = cv2.addWeighted(lena, 0.8,
                        lily, 0.2,
                        0)  # Brightness Adjustment
cv2.imshow('add_w', add_w)

cv2.waitKey()
cv2.destroyAllWindows()