from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
import random as rng
from skimage.measure import compare_ssim as ssim
import operator
import matplotlib.pyplot as plt
import glob
import os, errno, sys	
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
i = 0
for img in glob.glob("{:s}//{:s}".format(args.input,"c_*.png")):
    extracted.append([i,cv.imread(img,cv.IMREAD_UNCHANGED)])
    i += 1

img = extracted
n_imgs = len(extracted)
img_indexes = np.arange(0,n_imgs,1)

with open(args.params) as fp:
	params = fp.readlines()

n_size = int(params[0])
size = int(params[1])

generation_size = 100
n_best = 2
n_parents = int(generation_size/2) - n_best
generation = []
for i in range(generation_size):
	img = extracted.copy()
	random.shuffle(img) 
	generation.append(img)

(H, W) = image.shape[:2]

lattice_const = int(H/size)

pos = [[i*lattice_const,j*lattice_const] for i in range(size) for j in range(size)]

def scoreSubimage(image1, image2):
    score = (image1.mean() - image2.mean())**2 + sys.float_info.epsilon
    return 1/score

def meanRootSquareCmp(image, image2, n, m):
    if image.shape != image2.shape:
        return -1
    (W, H) = image[:, :, 0].shape
    score = 0
    for i in range(n):
        for j in range(m):
            score += scoreSubimage(image[lattice_const*i : lattice_const*(i+1), lattice_const*j : lattice_const*(j+1)],
                                   image2[lattice_const*i:lattice_const*(i+1), lattice_const*j:lattice_const*(j+1)])
    return score/(n*m)

def draw(img, show=False):
	output = np.zeros(image.shape,dtype=np.uint8)
	for i,p in zip(img,pos):
		output[p[0]:p[0]+lattice_const,p[1]:p[1]+lattice_const] = i[1]
	if show:
		cv.imshow("img",output)		
		cv.waitKey(1)
	return output

def weighted_random_choice(fitness):
    max = sum(fitness.values())
    pick = np.random.uniform(0, max)
    current = 0
    for key, value in fitness.items():
        current += value
        if current > pick:
            return key

def get_best(generation,fitness):
    sorted_fitness = sorted(fitness, key=fitness.get)
    best = [generation[i] for i in sorted_fitness[len(sorted_fitness) - n_best:]]
    return best

def order1_crossover(parent1, parent2):
	cross = np.random.randint(low=0,high=n_imgs,size=2)
	while cross[0] == cross[1]:
		cross = np.random.randint(low=0,high=n_imgs,size=2)
	child = [[-1,np.zeros((lattice_const,lattice_const,4))] for i in range(n_imgs)]
	child[min(cross):max(cross)] = parent1[min(cross):max(cross)]
	child_inds = [ch[0] for ch in child]
	missing_pictures = [i for i in img_indexes if i not in child_inds]
	missing_pictures.sort()
	parent2_inds = [p[0] for p in parent2]
	child[:min(cross)] = [parent2[parent2_inds.index(missing_pictures[i])] for i in range(0,min(cross))]
	child[max(cross):] = [parent2[parent2_inds.index(missing_pictures[i+min(cross)])] for i in range(0,n_imgs-max(cross))]
	return child

def single_swap_mutation(child):
	swap = np.random.randint(low=0,high=n_imgs,size=2)
	while swap[0] == swap[1]:
		swap = np.random.randint(low=0,high=n_imgs,size=2)
	child[swap[0]], child[swap[1]] = child[swap[1]], child[swap[0]]
	return child

def new_generation(parents):
    children = []
    mutation_p = 0.05
    child_p = 0.7
    p = np.random.rand(2,n_parents + n_best) 
    for i in range(n_parents + n_best):
        indexes = np.random.randint(low=0,high=n_parents,size=2)
        while indexes[0] == indexes[1]:
        	indexes = np.random.randint(low=0,high=n_parents,size=2)
       	img1 = parents[indexes[0]]
        img2 = parents[indexes[1]]

        child = order1_crossover(img1, img2)
        if p[0][i] < mutation_p:
        	child = single_swap_mutation(child)

        if p[1][i] < child_p:
        	children.append(child)
        else:
        	children.append(img1)
    return parents + children

def fitness_func(score,i):
    return score*((i+1)/1000) + 1

def fitness(generation, i):
    n_fitness = {}
    fitness = {}
    for g in range(len(generation)):
        # score = ssim(image,draw(generation[g]),multichannel=True)
        score = meanRootSquareCmp(image, draw(generation[g],True), n=size, m=size)
        n_fitness[g] = fitness_func(score,i)
        fitness[g] = score
    return fitness, n_fitness


iterations = []
best_score = []
average_score = []
it = 0
try:
	while True:
	    it += 1
	    print("i={:d}".format(it))
	    (fit, n_fit) = fitness(generation_size,i)
	    output = draw(generation[max(fit.items(), key=operator.itemgetter(0))[0]])
	    cv.imwrite("{:s}/out_{:d}.png".format(args.output,it), output)   
	    best_score.append(max(fit.items(), key=operator.itemgetter(1))[1])
	    iterations.append(it)
	    average_score.append(sum(fit.values())/(len(fit.values())))
	    parents = [generation[weighted_random_choice(n_fit)] for i in range(n_parents)]
	    n_generation = new_generation(parents)   
	    generation = n_generation + get_best(generation,n_fit)
	    print(len(generation))
except KeyboardInterrupt:
    pass

plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
plt.plot(iterations,best_score,'-r',label="best")
plt.plot(iterations,average_score,'-b',label="average")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("score")
plt.savefig("{:s}/plot.png".format(args.output))