import numpy as np
import cv2

def order_points(points):
	rect = np.zeros((4,2),dtype="float32")
	'''
	0-------1
	|		|	
	|		|
	|		|
	3-------2
	'''
	#getting sum of rows array
	s = points.sum(axis=1)
	# max sum is bottom right and min sum is top left
	rect[0] = points[np.argmin(s)]
	rect[2] = points[np.argmax(s)]

	#getting difference
	s = np.diff(points,axis=1)
	# max diff is bottom left corner and min diff is top right
	rect[1] = points[np.argmin(s)]
	rect[3] = points[np.argmax(s)]

	return rect

def fourpoint_transform(img,points):
	rect = order_points(points)
	(tl,tr,br,bl) = rect

	#width of new image will be maximum of distance between
	#bottom right and bottom left or top right and top left
	width1 = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
	width2 = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))
	maxWidth = max(int(width1),int(width2))

	#similarly height of new image
	h1 = np.sqrt(((bl[0]-tl[0])**2)+((bl[1]-tl[1])**2))
	h2 = np.sqrt(((br[0]-tr[0])**2)+((br[1]-tr[1])**2))
	maxHeight = max(int(h1),int(h2))

	#now we have dimensions of new image
	#constructing set of destination points

	des = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]],dtype="float32")
	M = cv2.getPerspectiveTransform(rect,des)
	warped = cv2.warpPerspective(img,M,(maxWidth,maxHeight))

	return warped