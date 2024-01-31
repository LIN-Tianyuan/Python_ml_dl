"""
Zooming in and out of images
"""
import cv2

img = cv2.imread('../data/Linus.png')
cv2.imshow('img', img)
# reduce
h, w = img.shape[:2]
dst_size = (int(w/2), int(h/2))
reduce = cv2.resize(img, dst_size)
cv2.imshow('reduce', reduce)
# enlarge
dst_size = (w*2, h*2)
nearest = cv2.resize(img, dst_size,
                     interpolation=cv2.INTER_NEAREST)
cv2.imshow('nearest', nearest)

# bilinear interpolation
linear = cv2.resize(img, dst_size,
                    interpolation=cv2.INTER_LINEAR)
cv2.imshow('linear', linear)
cv2.waitKey()
cv2.destroyAllWindows()