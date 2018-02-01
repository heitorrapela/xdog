# XDOG-image-stylization-test

Simple implementation of XDOG for studies 


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

	So opencv cv2.GaussianBlur is faster and


References used in these project

[XDoG: An eXtended difference-of-Gaussians compendium including advanced image stylization](http://www.kyprianidis.com/p/cag2012/winnemoeller-cag2012.pdf)
[garygrossi/XDoG-Python](https://github.com/garygrossi/XDoG-Python)