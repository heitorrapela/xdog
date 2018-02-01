import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import time
from matplotlib import pyplot as plt

def calculateHistogram3d(img):
	color = ('b','g','r')
	img = abs(img)
	for i,col in enumerate(color):
	    histr = cv2.calcHist([img],[i],None,[256],[0,256])
	    plt.plot(histr,color = col)
	    plt.xlim([0,256])
	plt.show()

def calculateHistogram1d(img):
	plt.hist(img.ravel(),256,[0,256]); plt.show()

# Difference of Gaussians applied to img input
def dog(img,size=(0,0),k=1.6,sigma=0.5,gamma=1):
	img1 = cv2.GaussianBlur(img,size,sigma)
	img2 = cv2.GaussianBlur(img,size,sigma*k)
	return (img1-gamma*img2)

# Threshold the dog image, with dog(sigma,k) > 0 ? 1(255):0(0)
def edge_dog(img,sigma=0.5,k=200,gamma=0.98):
	aux = dog(img,sigma=sigma,k=k,gamma=0.98)
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] > 0):
				aux[i,j] = 255
			else:
				aux[i,j] = 0
	return aux

# xdog adapted from garygrossi
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

# version of xdog inspired by article
def xdog(img,sigma=0.5,k=1.6, gamma=1,epsilon=1,phi=1):
	aux = dog(img,sigma=sigma,k=k,gamma=gamma)/255
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] < epsilon):
				aux[i,j] = 1*255
			else:
				aux[i,j] = 255*(1 + np.tanh(phi*(aux[i,j])))
	return aux


if __name__ == '__main__':
	# Open image in grayscale
	#img = cv2.imread('imgs/rapela.jpg',cv2.CV_LOAD_IMAGE_GRAYSCALE)
	
	img = cv2.imread('imgs/lena.png',cv2.CV_LOAD_IMAGE_GRAYSCALE)

	# k = 1.6 as proposed in the paper
	k = 1.6

	#cv2.imshow("Lena", dog(img))
	#edge_dog(img)
	#cv2.imshow("tessst",edge_dog(img,sigma=0.5,k=200, gamma=0.98))
	#cv2.waitKey(0)
	#cv2.imshow("Lena",dog(img,gamma=1))
	#cv2.waitKey(0)
	cv2.imshow("XBUDOG",np.uint8(xdog_garygrossi(img,sigma=0.5,k=200, gamma=0.98,epsilon=0.1,phi=10)))
	#cv2.imshow("XBUDOG2",np.uint8(xdog_garygrossi(img)))
	cv2.imshow("XBUDOG_mine",np.uint8(xdog(img,sigma=0.4,k=1.6, gamma=0.5,epsilon=-0.5,phi=10)))
	
	# Test
	cv2.imshow("XBUDOG_mineTest",np.uint8(xdog(img,sigma=1.6,k=1.6, gamma=0.5,epsilon=-1,phi=10)))

	# Natural media (tried to follow parameters of article)
	cv2.imshow("Natural Media",np.uint8(xdog(img,sigma=1,k=1.6, gamma=0.5,epsilon=-0.5,phi=10)))

	cv2.waitKey(0)

'''
# Video version - Its not real time version :)
if __name__ == '__main__':
	# Open image in grayscale
	cap = cv2.VideoCapture(0)
		
	# k = 1.6 as proposed in the paper
	k = 1.6

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Display the resulting frame
		cv2.imshow("Natural Media",np.uint8(xdog(gray,sigma=1,k=1.6, gamma=0.5,epsilon=-0.5,phi=10)))
		cv2.imshow("Gary",np.uint8(xdog_garygrossi(gray)))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
'''