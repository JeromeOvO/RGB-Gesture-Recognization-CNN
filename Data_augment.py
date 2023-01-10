import numpy as np
import cv2
import imutils

image = cv2.imread(r"D:\Undergraduate\pic\5.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_original = image
image_original = cv2.resize(image_original, (300, 300))
cv2.imshow('original', image_original)
image = cv2.bilateralFilter(image, 9, 75, 75)
image = cv2.resize(image, (300, 300))
cv2.imshow('test', image)
cv2.waitKey(0)
cv2.destroyAllWindows()