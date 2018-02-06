import cv2
import numpy as np
import time
import numpy as np
from numba import jit

# Difference of Gaussians applied to img input
@jit
def dog(img,size=(0,0),k=1.6,sigma=0.5,gamma=1):
	img1 = cv2.GaussianBlur(img,size,sigma)
	img2 = cv2.GaussianBlur(img,size,sigma*k)
	return (img1-gamma*img2)

# garygrossi xdog version
@jit
def xdog_garygrossi(img,sigma=0.5,k=200, gamma=0.98,epsilon=0.1,phi=10):
	aux = dog(img,sigma=sigma,k=k,gamma=gamma)/255
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] >= epsilon):
				aux[i,j] = 1
			else:
				ht = np.tanh(phi*(aux[i][j] - epsilon))
				aux[i][j] = 1 + ht
	return aux*255

@jit
def hatchBlend(image):
	xdogImage = xdog(image,sigma=1,k=200, gamma=0.5,epsilon=-0.5,phi=10)
	hatchTexture = cv2.imread('./imgs/hatch.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
	hatchTexture = cv2.resize(hatchTexture,(image.shape[1],image.shape[0]))
	alpha = 0.120
	return (1-alpha)*xdogImage + alpha*hatchTexture

# version of xdog inspired by article
@jit
def xdog(img,sigma=0.5,k=1.6, gamma=1,epsilon=1,phi=1):
	aux = dog(img,sigma=sigma,k=k,gamma=gamma)/255
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] < epsilon):
				aux[i,j] = 1*255
			else:
				aux[i,j] = 255*(1 + np.tanh(phi*(aux[i,j])))
	return aux


# Video version - Its not real time version :)
if __name__ == '__main__':
	# Open image in grayscale
	cap = cv2.VideoCapture(0)
		
	# k = 1.6 as proposed in the paper
	k = 1.6

	# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	hatchTexture = cv2.imread('./imgs/hatch.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
	
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		original = frame
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		cv2.imshow("Original",np.uint8(original))

		cv2.imshow("XDoG GaryGrossi",np.uint8(xdog_garygrossi(frame,sigma=0.5,k=200, gamma=0.98,epsilon=0.1,phi=10)))

		cv2.imshow("XDoG Project 1",np.uint8(xdog(frame,sigma=0.4,k=1.6, gamma=0.5,epsilon=-0.5,phi=10)))

		cv2.imshow("XDoG Project 2",np.uint8(xdog(frame,sigma=1.6,k=1.6, gamma=0.5,epsilon=-1,phi=10)))

		# Natural media (tried to follow parameters of article)
		cv2.imshow("XDoG Project 3 - Natural Media",np.uint8(xdog(frame,sigma=1,k=1.6, gamma=0.5,epsilon=-0.5,phi=10)))

		cv2.imshow("XDoG Project 4 - Hatch",np.uint8(hatchBlend(frame)))

		#cv2.imshow("Natural Media",np.uint8(xdog(gray,sigma=1,k=1.6, gamma=0.5,epsilon=-0.5,phi=10)))
		cv2.imshow("Natural Media",np.uint8(frame))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()