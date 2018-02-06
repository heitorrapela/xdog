# python-xdog

## Simple implementation of XDoG

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
<img width="400" alt="rapela" src="https://github.com/heitorrapela/xdog/blob/master/imgs/rapela.jpg"> a) Original | <img width="400" alt="rapela_grayscale" src="https://github.com/heitorrapela/xdog/blob/master/imgs/original_grayscale.jpg"> b) Grayscale | <img width="400" alt="xdog_garygrossi" src="https://github.com/heitorrapela/xdog/blob/master/imgs/xdog_garygrossi.jpg"> c) Xdog GaryGrossi
<img width="400" alt="xdog_nat" src="https://github.com/heitorrapela/xdog/blob/master/imgs/xdog_naturalMedia.jpg"> d) Xdog Test Nat | <img width="400" alt="xdog_project1" src="https://github.com/heitorrapela/xdog/blob/master/imgs/xdog_project1.jpg"> e) Xdog Test | <img width="400" alt="xdog_hatch" src="https://github.com/heitorrapela/xdog/blob/master/imgs/xdog_hatch.jpg"> f) Xdog Hatch


----------


**Parameters**

 - a) Original image RGB 
 - b) Load as grayscale (OpenCV) 
 - c) XdogGary: sigma=0.5, k=200, gamma=0.98, epsilon=0.1, phi=10 
 - d) Xdog this project: sigma=1, k=1.6, gamma=0.5, epsilon=-0.5, phi=10
 - e) Xdog this project: sigma=0.4, k=1.6, gamma=0.5, epsilon=-0.5, phi=10
 - f)  Xdog this project: hatchBlend function

----------
## Dependencies

 - Python 2.7 
 - OpenCV (Tested on 2.4.13)

		sudo apt-get install python-opencv
		
 - Numpy 1.11.2
 
		pip install numpy

If you want to run video_jit.py:
 - [Numba](https://numba.pydata.org/)
  
		conda install numba

There is a requirement.txt to install numpy and numba:
	
	pip install -r requirements.txt


----------


**Test time in Difference of Gaussians (DoG) implementation**

	start = time.time()
	img1 = cv2.GaussianBlur(img,(0,0),0.5)
	elapsed = time.time() - start
	elapsed = 0.000515937805176

	start = time.time()	
	sigma = 0.5
	gauss1 = gaussian_filter(img, 0.5)
	elapsed = time.time() - start
	print elapsed

	elapsed = 0.00245594978333

	So opencv cv2.GaussianBlur is faster. The difference from gaussian blur results from OpenCV and SciPy is similar.
	The difference from each mat can be seen in image below:

![Difference from mat of cv2.GaussianBlur and gaussian_filter](https://github.com/heitorrapela/xdog/blob/master/imgs/gaussian_opencvXscipy.png)


## References used in these project

[Xdog Article](http://www.kyprianidis.com/p/cag2012/winnemoeller-cag2012.pdf)

[garygrossi/XDoG-Python](https://github.com/garygrossi/XDoG-Python)
