from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
import random as rng
# import imutils
# from skimage.measure import compare_ssim as ssim
import operator
import matplotlib.pyplot as plt
import glob
import os, errno
import random

parser = argparse.ArgumentParser(description='ga')
parser.add_argument('--input', help='Name of input folder', default='./input')
parser.add_argument('--output', help='Name of output folder', default='./output')
parser.add_argument('--params', help='Params file', default='params.txt')

args = parser.parse_args()

try:
    os.makedirs(args.output)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

for img in glob.glob("{:s}//{:s}".format(args.output,"*.png")):
    os.remove(img)

image = cv.imread("{:s}//{:s}".format(args.input,"input_resized.png"),cv.IMREAD_UNCHANGED)
cv.imwrite("{:s}/input.png".format(args.output), image)   
    
extracted = []
for img in glob.glob("{:s}//{:s}".format(args.input,"c_*.png")):
    extracted.append(cv.imread(img,cv.IMREAD_UNCHANGED))

with open(args.params) as fp:
	params = fp.readlines()

n_size = int(params[0])
size = int(params[1])

(H, W) = image.shape[:2]

lattice_const = int(H/size)

pos = [[i*lattice_const,j*lattice_const] for i in range(size) for j in range(size)]
img = extracted
output = np.zeros(image.shape,dtype=np.uint8)

population_size = 10
generation = []
for i in range(population_size):
	img = extracted.copy()
	random.shuffle(img) 
	generation.append(img)

def draw(img):
	output = np.zeros(image.shape,dtype=np.uint8)
	for i,p in zip(img,pos):
		output[p[0]:p[0]+lattice_const,p[1]:p[1]+lattice_const] = i
	cv.imshow("img",output)		
	cv.waitKey(0)

for img in generation:
	draw(img)