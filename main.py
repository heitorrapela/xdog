import cv2
import numpy as numpy
k = 1.6

# For default value: sigma2*k = 0
def dog(img,size1=(7,7),size2=(5,5),sigma1=0.5,sigma2=0.5,theta=1):

	if(size1[0] < size2[0]):
		aux = size1
		size1 = size2
		size2 = aux
	img1 = cv2.GaussianBlur(img,size1,sigma1)
	img2 = cv2.GaussianBlur(img,size2,sigma2*k) 
	return (img1-theta*img2)

def edge_dog(img):
	aux = dog(img)
	teste = aux

	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] > 0):
				teste[i,j] = 255
			else:
				teste[i,j] = 0
	return teste

#def xdog(img,size1=(7,7),size2=(5,5),sigma1=0,sigma2=0):


if __name__ == '__main__':
	# Open image in grayscale
	img = cv2.imread('imgs/lena.png',0)
	#cv2.imshow("Lena", dog(img))
	edge_dog(img)
	cv2.imshow("tessst",edge_dog(img))
	cv2.waitKey(0)
	cv2.imshow("Lena",dog(img,theta=1))
	cv2.waitKey(0)