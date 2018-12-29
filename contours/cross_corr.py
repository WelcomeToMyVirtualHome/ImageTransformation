from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
from matplotlib import pyplot as plt
from scipy import signal
import glob


def cross_corr_plot(img1, img2):
	corr = signal.correlate2d(img1, img2, boundary='symm', mode='same')
	y, x = np.unravel_index(np.argmax(corr), corr.shape)

	fig, (ax_orig, ax_template, ax_corr) = plt.subplots(1, 3)
	ax_orig.imshow(img1, cmap='gray')
	ax_orig.set_title('Original')
	ax_orig.set_axis_off()
	ax_template.imshow(img2, cmap='gray')
	ax_template.set_title('Template')
	ax_template.set_axis_off()
	ax_corr.imshow(corr, cmap='gray')
	ax_corr.set_title('Cross-correlation')
	ax_corr.set_axis_off()
	ax_orig.plot(x, y, 'ro')
	plt.show()

parser = argparse.ArgumentParser(description='ga')
parser.add_argument('--input', help='Name of input folder', default='./input')
args = parser.parse_args()

image = cv.imread("{:s}//{:s}".format(args.input,"input_resized.png"),cv.IMREAD_GRAYSCALE)
    
extracted = []
for img in glob.glob("{:s}//{:s}".format(args.input,"c_*.png")):
    extracted.append(cv.imread(img,cv.IMREAD_GRAYSCALE))

cross_corr_plot(image, extracted[np.random.randint(0,len(extracted))])