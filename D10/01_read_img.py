'''
Read, Display, Save
'''

import cv2

# Read image
img = cv2.imread('../data/Linus.png')
print(img.shape)    # (216, 160, 3) Height, Width, Channel

# Show image
cv2.imshow('img', img)

# Save Image
cv2.imwrite('./new_linus.png', img)
# Actively enters a blocking state and waits for the user to press a key
cv2.waitKey()
cv2.destroyAllWindows()
