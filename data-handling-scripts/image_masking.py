import cv2
import numpy as np

img = cv2.imread('162.png')

# Invert and convert to HSV
img_hsv = cv2.cvtColor(255-img, cv2.COLOR_BGR2HSV)
img_hsv_g = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])

mask = cv2.inRange(img_hsv, low_green, high_green)
mask_g = cv2.inRange(img_hsv_g, low_green, high_green)

# Inpaint red box
result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
result = cv2.inpaint(result, mask_g, 3, cv2.INPAINT_TELEA)

cv2.imshow('2', result)
cv2.waitKey(0)

cv2.imwrite('test.png', result)