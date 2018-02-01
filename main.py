import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import time
from matplotlib import pyplot as plt

def timed(f):
  start = time.time()
  ret = f()
  elapsed = time.time() - start
  return ret, elapsed

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
# Debug
def dog(img,size=(0,0),k=1.6,sigma=0.5,gamma=1):
	start = time.time()
	img1 = cv2.GaussianBlur(img,size,sigma)
	elapsed = time.time() - start
	print elapsed
	img1 = cv2.GaussianBlur(img,size,sigma)
	img2 = cv2.GaussianBlur(img,size,sigma*k)

	start = time.time()	
	gauss1 = gaussian_filter(img, sigma)
	elapsed = time.time() - start
	print elapsed

	print "GAUSS1: ", gauss1[0][0]
	gauss2 = gamma*gaussian_filter(img, sigma*k)
	print gauss1
	differenceGauss = gauss1 - gauss2
	#return differenceGauss

	print type(img1), img.shape, np.count_nonzero(img1)
	print type(gauss1), gauss1.shape, np.count_nonzero(gauss1)
	ret = img1-gauss1
	print np.count_nonzero(ret)
	print ret.shape
	calculateHistogram1d(ret)
	return (img1-gamma*img2)

# Threshold the dog image, with dog(sigma,k) > 0 ? 1(255):0(0)
'''
def edge_dog(img,sigma=0.5,k=1.6):
	aux = dog(img,sigma=sigma,k=k)
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] > 0):
				aux[i,j] = 255
			else:
				aux[i,j] = 0
	return aux
'''

def xdog(img,sigma=0.5,k=1.6, gamma=1,epsilon=1,phi=1):
	print sigma, k , gamma, phi
	aux = dog(img,sigma=sigma,k=k,gamma=gamma)/255
	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] >= epsilon):
				aux[i,j] = 1
			else:
				#aux[i,j] = 255 + np.tanh(phi*(aux[i,j]))
				ht = np.tanh(phi*(aux[i][j] - epsilon))
				aux[i][j] = 1 + ht
	return aux*255

if __name__ == '__main__':
	# Open image in grayscale
	img = cv2.imread('imgs/rapela.jpg',cv2.CV_LOAD_IMAGE_GRAYSCALE)
		
	# k = 1.6 as proposed in the paper
	k = 1.6

	#cv2.imshow("Lena", dog(img))
	#edge_dog(img)
	#cv2.imshow("tessst",edge_dog(img))
	#cv2.waitKey(0)
	#cv2.imshow("Lena",dog(img,gamma=1))
	#cv2.waitKey(0)
	cv2.imshow("XBUDOG",np.uint8(xdog(img,sigma=0.5,k=200, gamma=0.98,epsilon=0.1,phi=10)))
	#result = xdog(img)
	#cv2.imshow("Image",np.uint8(result))
	cv2.waitKey(0)
	#cv2.imwrite("res.jpg", result)