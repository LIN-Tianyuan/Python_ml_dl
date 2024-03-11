"""
Find and draw outlines
"""
import cv2
img = cv2.imread('../data/3.png')
# cv2.imshow('img', img)
# grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# binarization
t, binary = cv2.threshold(gray,
                          127,
                          255,
                          cv2.THRESH_BINARY)
# Find Outline
cnts, hie = cv2.findContours(binary,    # image
                             cv2.RETR_EXTERNAL, # Detects only the outer contour
                             cv2.CHAIN_APPROX_NONE) # Save all coordinate points
# print(len(cnts))
# for i in cnts:
#     print(i.shape)
# print(hie)

# outline
res = cv2.drawContours(img,  # On which image to draw
                       cnts, # contour coordinates
                       -1,   # Which outline to draw
                       (0, 0, 255), # Color
                       2)    # Thickness of the line
cv2.imshow('res', res)
cv2.waitKey()
cv2.destroyAllWindows()