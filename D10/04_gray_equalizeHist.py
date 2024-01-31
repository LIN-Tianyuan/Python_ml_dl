"""
Histogram equalization for grayscale images
"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../data/sunrise.jpg', 0)
cv2.imshow('img', img)

# equalization
res = cv2.equalizeHist(img)
cv2.imshow('res', res)

# bar chart
plt.subplot(2, 1, 1)
plt.hist(img.ravel(),
         bins=256,
         range=(0,256))
plt.subplot(2, 1, 2)
plt.hist(res.ravel(),
         bins=256,
         range=(0,256))
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()