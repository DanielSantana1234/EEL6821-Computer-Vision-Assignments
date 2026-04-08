"""
IV.   Principal Component Analysis
This homework project allows to implement and understand the merits of the PCA on multispectral images.  
This document shows the results of applying PCA algorithm over multispectral satellite Images. 
In particular, we selected an 8-band image (i.e. coastal, blue, green, yellow, red, red edge, NIR1 and NIR2); 
see below their respective wavelength intervals, in which we can see different geographic properties like 
a river, forest (trees), building structures, green space, etc. This image is 16-bit depth and its location 
is Adelaide, Australia. Please see below a representation of the RGB channels of the 8-band image.
Spectral bands considered include:
Coastal band [400–450 nm],
Blue [450–510 nm],
Green (510–580 nm)
Yellow [585–625 nm],
Red [630–690 nm]
Red-edge [705–745 nm]
Near-infrared 1, NIR1 [770–895 nm]
Near-infrared 2, NIR2; [860–1040 nm]
Given these 8-band image, construct a matrix of, where    and  and  are 
the corresponding width and height of the input image, with  representing the 8 dimensions or wavebands. 
Hence the rows of this matrix correspond to observations and columns correspond to variables.
Compute the covariance matrix
Determine the eigenvalues and eigenvectors (principal components) of the covariance matrix.
Through the inverse PCA, reconstruct the original image by keeping
each of the Principal Component (PC) starting from the first to the eight PC and see what you get in 
each of these reconstructed mages.
Reconstruct the input image from the eigenvectors of the highest 4 eigenvalues
Reconstruct the input image from the eigenvectors of the lowest 4 eigenvalues
Subtract the original image from the reconstructed image you obtained in (b)
Subtract the original image from the reconstructed image you obtained in (c)
Provide your assessment or concluding remarks on these reconstructed images and on the subtractions performed
"""
