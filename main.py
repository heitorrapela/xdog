import cv2
import numpy as np

# Difference of Gaussians applied to img input
def dog(img,size=(5,5),k=1.6,sigma=0.5,theta=1):
	img1 = cv2.GaussianBlur(img,size,sigma)
	img2 = cv2.GaussianBlur(img,size,sigma*k) 
	return (img1-theta*img2)

# Threshold the dog image, with dog(sigma,k) > 0 ? 1(255):0(0)
def edge_dog(img,sigma=0.5,k=1.6):
	aux = dog(img,sigma=sigma,k=k)
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] > 0):
				aux[i,j] = 255
			else:
				aux[i,j] = 0
	return aux

def xdog(img,sigma=0.5,k=1.6, theta=1,epsilon=1,phi=1):
	aux = dog(img,sigma=sigma,k=k)
	print type(aux)
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] < epsilon):
				aux[i,j] = 255
			else:
				aux[i,j] = 255 + np.tanh(phi*(aux[i,j]))
	return aux



if __name__ == '__main__':
	# Open image in grayscale
	img = cv2.imread('imgs/lena.png',0)
		
	# k = 1.6 as proposed in the paper
	k = 1.6

	#cv2.imshow("Lena", dog(img))
	#edge_dog(img)
	cv2.imshow("tessst",edge_dog(img))
	cv2.waitKey(0)
	cv2.imshow("Lena",dog(img,theta=1))
	cv2.waitKey(0)
	cv2.imshow("XBUDOG",xdog(img,sigma=0.5,k=1.6, theta=1,epsilon=1,phi=1))
	cv2.waitKey(0)