import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def edge_detect(img,threshold_ratio=0.3):
	'''
	For our edge detection algorithm, we will be using canny edge detection, coupled with Otsu's algorithm to determine proper
	threshold values for the cv2 canny edge detection.
	'''
	
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred_gray = cv2.GaussianBlur(gray_image,(5,5),0)
	ret,th = cv2.threshold(blurred_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	lower = int(max(0 ,(1-threshold_ratio)*ret))
	upper = int(min(255,(1+threshold_ratio)*ret))

	blurred_img = cv2.GaussianBlur(img,(5,5),0)
	edges = cv2.Canny(img,lower,upper)

	# cv2.imshow("canny_edges", edges)
	# cv2.waitKey(200)
	return edges

def create_clothing_mask(img):
	'''
	To generate the clothing mask, we will first seperate background and foreground from the image using canny edge detection.
	Following that, we will use the grabcut algorithm to further isolate the foreground figure. 
	From there, we can potentially go through iterations of edge detection and graph cut in order to seperate the foreground figure
	into distinctions between human and clothing.

	'''
	

	### Edge Detection
	edges = edge_detect(img)
	bool_edges = np.array(edges,dtype=bool)
	struct = scipy.ndimage.generate_binary_structure(2, 2) #include diagonals in binary structure for binary closing morphology

	initial_mask = scipy.ndimage.binary_closing((bool_edges),struct,iterations=30) #fill all holes best you can
	initial_mask = scipy.ndimage.binary_erosion((initial_mask),struct,iterations=2).astype(np.uint8)
	initial_mask[initial_mask==0] = 2
	initial_mask[initial_mask==1] = 3 #Marking all initial mask pixels as possible foreground for grabcut
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	
	

	mask, bgdModel, fgdModel = cv2.grabCut(img,initial_mask,None,bgdModel,fgdModel,25,cv2.GC_INIT_WITH_MASK)
	#Change all possible-tagged values to respective foreground or background
	mask = np.where((mask==2)|(mask==0),0,1).astype('uint8') 
	# print(set((mask.flatten()).tolist()))
	cv2.imshow("binary", img*mask[:,:,np.newaxis])
	cv2.waitKey(200)
	'''
	NOTE: This currently only returns the initial mask of the entire model. We still need to remove the human features itself.
	'''

	return mask