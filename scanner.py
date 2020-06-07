#Usage python sacnner.py --image images/page.jpg 
from transformer.transform import fourpoint_transform
from skimage.filters import threshold_local
import numpy as nu
import argparse
import cv2
import imutils

#getting command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to image to scan")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0]/500.0
orig = image.copy()
image = imutils.resize(image,height = 500)

#converting to grayscale,blur it and finding edges
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(gray,75,200)

#showing original and edge detected images
print("STEP1: Edge detection")
cv2.imshow("Image",image)
cv2.imshow("Edged",edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

#finding contours in the edged image

cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts,key = cv2.contourArea,reverse = True)[:5]

#loop over contours
for c in cnts:
	#approximate the contour
	peri = cv2.arcLength(c,True)
	approx = cv2.approxPolyDP(c,0.02*peri,True)

	if len(approx) == 4:
		screenCnt = approx
		break

print("STEP 2: Find contours of paper")
cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
cv2.imshow("Outline",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#applying four point transform

warped = fourpoint_transform(orig,screenCnt.reshape(4,2)*ratio)

warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
T = threshold_local(warped,11,offset = 10,method = "gaussian")
warped = (warped>T).astype("uint8")*255

print("STEP 3: Apply perspective transform")
cv2.imshow("Original",imutils.resize(orig,height = 650))
cv2.imshow("Scanned", imutils.resize(warped,height = 650))
name = "scanned_"+args["image"];
cv2.imwrite(name,imutils.resize(warped,height=650));
cv2.waitKey(0)
cv2.destroyAllWindows()