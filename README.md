# python-xdog

## Simple implementation of XDOG for studies 

Test time in Difference of Gaussians (DoG) implementation:

	start = time.time()
	size = (0,0)
	sigma = 0.5
	img1 = cv2.GaussianBlur(img,size,sigma)
	elapsed = time.time() - start
	elapsed = 0.000515937805176

	start = time.time()	
	sigma = 0.5
	gauss1 = gaussian_filter(img, sigma)
	elapsed = time.time() - start
	print elapsed

	elapsed = 0.00245594978333

	So opencv cv2.GaussianBlur is faster. The difference from gaussian blur result from OpenCV and SciPy is similar.
	The difference from each mat can be seen in image below (image in folder /imgs/gaussian_opencvXscipy.png):

UPDATE IMAGE LINK

## References used in these project

[XDoG Article](http://www.kyprianidis.com/p/cag2012/winnemoeller-cag2012.pdf)

[garygrossi/XDoG-Python](https://github.com/garygrossi/XDoG-Python)