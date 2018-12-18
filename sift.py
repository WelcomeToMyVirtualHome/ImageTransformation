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

gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)


kp1, des1 = sift.detectAndCompute(image1,None)
kp2, des2 = sift.detectAndCompute(image2,None)

img = cv.drawKeypoints(gray1,kp1,image1)
cv.imshow("img",img)
cv.waitKey(0)

img = cv.drawKeypoints(gray2,kp2,image2)
cv.imshow("img",img)
cv.waitKey(0)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

#the higher 'checks' value the higher precision
search_params = dict(checks=50)  

flann = cv.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv.drawMatchesKnn(image1,kp1,image2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()