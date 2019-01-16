from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
import os, errno
import glob
from matplotlib import pyplot as plt
import pandas as pd
from os import path

parser = argparse.ArgumentParser(description='Plot fitness')
parser.add_argument('--data', help='Path to file with fitness history', default='fintessHistory.dat')
parser.add_argument('--output', help='Output filename', default='./output/fintess.png')
args = parser.parse_args()

file_path = path.relpath(args.data)
height = 5
width = 10
opacity = 1
data = pd.read_table(file_path, sep=" ",header=None)
X = data[data.columns[0]].values
averageY = data[data.columns[1]].values
bestY = data[data.columns[2]].values
plt.figure(num=None, figsize=(width, height), dpi=80, facecolor='w', edgecolor='k')
plt.plot(X[1:],bestY[1:],"-r", alpha=opacity, label="Best")
plt.plot(X,averageY,"-b", alpha=opacity, label="Average")
plt.ylim(bottom=0)
plt.title("Fitness")
plt.ylabel("MSE")
plt.xlabel("Generation")
plt.legend()
plt.tight_layout()
plt.savefig(args.output)