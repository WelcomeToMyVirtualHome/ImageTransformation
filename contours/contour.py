from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
from matplotlib import pyplot as plt

sift = cv.xfeatures2d.SIFT_create()

parser = argparse.ArgumentParser(description='sift')
parser.add_argument('--img1', help='Name of first image', default='shapes.png')
parser.add_argument('--img2', help='Name of second image', default='n_shapes.png')
args = parser.parse_args()

image1 = cv.imread(args.img1,cv.IMREAD_UNCHANGED)
image2 = cv.imread(args.img2,cv.IMREAD_UNCHANGED)

if image1 is None or image2 is None:
	print("Couldn't load images")
	exit()

def reshape_extract(image):
	height = 255
	(h,w) = image.shape[:2]
	scale = float(height)/h
	n_h,n_w = height, image.shape[0]*scale
	image1 = cv.resize(image,(int(n_h),int(n_w)))

	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
	closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
	closing = cv.GaussianBlur(closing, (3, 3), 0)
	edged = cv.Canny(closing, 50, 100)
	_, cnts, _ = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	return cnts

cnts1 = reshape_extract(image1)
cnts2 = reshape_extract(image2)

cnt0 = cnts1[0]
cnt1 = cnts2[1]
print("Same={:.2f}".format(cv.matchShapes(cnt0,cnt0,1,0.0)))
image1 = cv.drawContours(image1, [cnt0], -1, 255, -1)
cv.imshow("img",image1)
cv.waitKey(0)

print("Different={:.2f}".format(cv.matchShapes(cnt0,cnt1,1,0.0)))
image1 = cv.drawContours(image1, [cnt0], -1, 255, -1)
image2 = cv.drawContours(image2, [cnt1], -1, 255, -1)
cv.imshow("img",image1)
cv.waitKey(0)
cv.imshow("img",image2)
cv.waitKey(0)

#TODO
#find two most simmilar contours -> mutate towards the most similar 
