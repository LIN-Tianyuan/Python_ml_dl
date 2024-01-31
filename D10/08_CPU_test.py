"""
Handling CPU3.png using binarization
1.Change the dial area to white and the background to black

Paper.jpg
2.Turns the paper white and the background black
"""
import cv2

# img = cv2.imread('../data/CPU3.png')
img = cv2.imread('../data/paper.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

# binarization
t, binary = cv2.threshold(gray,
                          200,
                          255,
                          cv2.THRESH_BINARY)

cv2.imshow('binary', binary)
cv2.waitKey()
cv2.destroyAllWindows()