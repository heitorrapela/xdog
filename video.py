import cv2
import numpy as np
import time

# version of xdog inspired by article
def xdog(img,sigma=0.5,k=1.6, gamma=1,epsilon=255,phi=1):
	img1 = cv2.GaussianBlur(img,(0,0),sigma)
	img2 = cv2.GaussianBlur(img,(0,0),sigma*k)
	aux = (img1-gamma*img2)/255

	for i in range(0,aux.shape[0]):
		for j in range(0,aux.shape[1]):
			if(aux[i,j] < epsilon):
				aux[i,j] = 255
			else:
				aux[i,j] = 255*(1 + np.tanh(phi*(aux[i,j])))
	return aux

# Video version - Its not real time version :)
if __name__ == '__main__':
	# Open image in grayscale
	cap = cv2.VideoCapture(0)
		
	# k = 1.6 as proposed in the paper
	k = 1.6

	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	hatchTexture = cv2.imread('./imgs/hatch.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
	alpha = 0.5#0.120
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		gray = np.uint8((frame[:,:,0]+frame[:,:,1]+frame[:,:,2])/3.0)

		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		
		for (x,y,w,h) in faces:
			roi_color = frame[y:y+h, x:x+w]
			#xdogroi = np.uint8(xdog(roi_color[:,:,0],sigma=1,k=1.6, gamma=0.5,epsilon=-0.5,phi=10))
			#hatchTexture = cv2.resize(hatchTexture,xdogroi.shape)
			#xdogroi = (1-alpha)*xdogroi + alpha*hatchTexture

			#frame[y:y+h,x:x+w,0] = xdogroi
			#frame[y:y+h,x:x+w,1] = xdogroi
			#frame[y:y+h,x:x+w,2] = xdogroi
			cv2.imshow("Natural Media",np.uint8(frame))


			cv2.imshow('img',roi_color)
		
		#cv2.imshow("Natural Media",np.uint8(xdog(gray,sigma=1,k=1.6, gamma=0.5,epsilon=-0.5,phi=10)))
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()