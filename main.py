import cv2
import numpy as numpy

img = cv2.imread('imgs/lena.png')

cv2.imshow("Lena",img)
cv2.waitKey(0)