import cv2
import numpy as numpy

def dog(img,size1=(7,7),size2=(5,5),sigma1=0,sigma2=0):
	if(size1[0] < size2[0]):
		aux = size1
		size1 = size2
		size2 = aux
	img1 = cv2.GaussianBlur(img,size1,sigma1)
	img2 = cv2.GaussianBlur(img,size2,sigma2)
	return img1-img2

if __name__ == '__main__':
	# Open image in grayscale
	img = cv2.imread('imgs/lena.png',0)
	cv2.imshow("Lena",dog(img))
	cv2.waitKey(0)