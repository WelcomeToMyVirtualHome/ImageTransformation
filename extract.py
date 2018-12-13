from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
import os, errno

def extract(image, cnts, dest):
    print("Extracting...")
    i = 0
    for contour in cnts:
        (x, y, w, h) = cv.boundingRect(contour)
        transparent_copy = np.zeros(image.shape,dtype="uint8")
        copy = image.copy()
        cv.drawContours(copy, [contour], -1, (255,255,255,255), 0)
        for xx in range(copy.shape[1]):
            for yy in range(copy.shape[0]):
                if cv.pointPolygonTest(contour,(xx,yy),False) > 0:
                    transparent_copy[yy,xx] = copy[yy,xx]
        cv.imwrite("{:s}/c_{:d}.png".format(dest,i), transparent_copy)   
        i = i + 1
    print("...Done")

parser = argparse.ArgumentParser(description='Extract')
parser.add_argument('--image', help='Path to input image.png', default='shapes.png')
parser.add_argument('--folder', help='Path to folder for preprocessed images', default='./input')
args = parser.parse_args()
try:
    os.makedirs(args.folder)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

src = cv.imread(args.image)
if src is None:
    print('Could not open or find the image:', args.image)
    exit(0)

image = cv.imread(args.image,cv.IMREAD_UNCHANGED)

hight = 200
(h,w) = image.shape[:2]
scale = float(hight)/h
n_h,n_w = hight, image.shape[0]*scale
image = cv.resize(image,(int(n_h),int(n_w)))

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
closing = cv.GaussianBlur(closing, (3, 3), 0)
edged = cv.Canny(closing, 50, 100)
_, cnts, _ = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

cv.imwrite("{:s}/input_resized.png".format(args.folder), image)         
extract(image,cnts,args.folder)

