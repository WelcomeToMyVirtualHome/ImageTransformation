from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
import os, errno
import glob
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Extract')
parser.add_argument('--image', help='Path to input image.png', default='picasso1.png')
parser.add_argument('--ref', help='Path to reference image.png', default='picasso1.png')
parser.add_argument('--input', help='Path to folder for preprocessed images', default='./input')
parser.add_argument('--output', help='Path to folder main programme output', default='./output')
parser.add_argument('--params', help='Params file', default='params.txt')
args = parser.parse_args()

try:
    os.makedirs(args.output)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(args.input)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

for img in glob.glob("{:s}//{:s}".format(args.output,"*.png")):
    os.remove(img)

for img in glob.glob("{:s}//{:s}".format(args.input,"*.png")):
    os.remove(img)

with open(args.params) as fp:
	params = fp.readlines()

image = cv.imread(args.image,cv.IMREAD_UNCHANGED)
if image is None:
    print('Could not open or find the image:', args.image)
    exit(0)

ref = cv.imread(args.ref,cv.IMREAD_UNCHANGED)
if ref is None:
    print('Could not open or find the image:', args.ref)
    exit(0)

n_size = int(params[0])
size = int(params[1])
image = cv.resize(image,(int(n_size),int(n_size)))
ref = cv.resize(ref,(int(n_size),int(n_size)))
cv.imwrite("{:s}/input_resized.png".format(args.input), ref)   
   
lattice_const = int(n_size/size)

ind = 0
pos = []
for i in range(size):
	for j in range(size):
		pos.append([i*lattice_const,j*lattice_const])
		img = image[i*lattice_const:(i+1)*lattice_const,j*lattice_const:(j+1)*lattice_const]
		cv.imwrite("{:s}/c_{:d}.png".format(args.input,ind), img)
		ind += 1

extracted = []
for img in glob.glob("{:s}//{:s}".format(args.input,"c_*.png")):
    extracted.append(cv.imread(img,cv.IMREAD_UNCHANGED))