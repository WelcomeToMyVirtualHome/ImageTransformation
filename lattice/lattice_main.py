from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
import random as rng
# import imutils
from skimage.measure import compare_ssim as ssim
import operator
import matplotlib.pyplot as plt
import glob
import os, errno
import random
import six

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


def draw(img):
	output = np.zeros(image.shape,dtype=np.uint8)
	for i,p in zip(img,pos):
		output[p[0]:p[0]+lattice_const,p[1]:p[1]+lattice_const] = i
	# cv.imshow("img",output)		
	# cv.waitKey(0)
	return output

def weighted_random_choice(fitness):
    max = sum(fitness.values())
    pick = np.random.uniform(0, max)
    current = 0
    for key, value in fitness.items():
        current += value
        if current > pick:
            return key

def get_best(generation,fitness,number_of_best):
    sorted_fitness = sorted(fitness, key=fitness.get)
    best = [generation[i] for i in sorted_fitness[len(sorted_fitness) - number_of_best:]]
    return best

def new_generation(parents, size, number_of_best):
    children = []
    mutation_p = 0.05
    while True:
        len_imgs = len(parents)
        ind1 = np.random.randint(low=0,high=len_imgs)
        ind2 = np.random.randint(low=0,high=len_imgs)
        while ind1 == ind2:
            ind2 = np.random.randint(low=0,high=len_imgs)
        img1 = parents[ind1]
        img2 = parents[ind2]
        cross = np.random.randint(low=1,high=len(img1)-1)
        child = img1[:cross] + img2[cross:]

        if np.random.rand() < mutation_p:
        	indexes = np.random.randint(low=0,high=len(img1),size=2)
        	child[indexes[0]], child[indexes[1]] = child[indexes[1]], child[indexes[0]]

        children.append(child)

        if len(parents) + len(children) == size - number_of_best:
            break
        
        cross = np.random.randint(low=1,high=len(img1)-1)
        child = img2[:cross] + img1[cross:]
        
        if np.random.rand() < mutation_p:
        	indexes = np.random.randint(low=0,high=len(img1),size=2)
        	child[indexes[0]], child[indexes[1]] = child[indexes[1]], child[indexes[0]]

        children.append(child)   
        if len(parents) + len(children) == size - number_of_best:
            break

    return parents + children

def fitness_func(score,i):
    # return some score function
    return score**((i+1)/100) + 10

def fitness(generation, i):
    n_fitness = {}
    fitness = {}
    for g in range(len(generation)):
        score = ssim(image,draw(generation[g]),multichannel=True)
        n_fitness[g] = fitness_func(score,i)
        fitness[g] = score
    return fitness, n_fitness

img = extracted
generation_size = 200
number_of_best = 2
generation = []
for i in range(generation_size):
	img = extracted.copy()
	random.shuffle(img) 
	generation.append(img)

num_iterations = 20
iterations = np.arange(start=0,stop=num_iterations,step=1)
best_score = []
average_score = []
for i in iterations:
    print("i={:d}".format(i))
    (fit, n_fit) = fitness(generation,i)
    output = draw(generation[max(fit.items(), key=operator.itemgetter(0))[0]])
    cv.imwrite("{:s}/out_{:d}.png".format(args.output,i), output)   
    best_score.append(max(fit.items(), key=operator.itemgetter(1))[1])
    average_score.append(sum(fit.values())/(len(fit.values())))
    parents = [generation[weighted_random_choice(n_fit)] for i in range(int(generation_size/2) - number_of_best)]
    n_generation = new_generation(parents,generation_size,number_of_best)   
    generation = n_generation + get_best(generation,n_fit,number_of_best)

plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
plt.plot(iterations,best_score,'-r',label="best")
plt.plot(iterations,average_score,'-b',label="average")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("ssim")
plt.savefig("{:s}/plot.png".format(args.output))