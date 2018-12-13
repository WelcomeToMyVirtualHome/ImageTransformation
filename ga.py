from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
import random as rng
import imutils
from skimage.measure import compare_ssim as ssim
import operator
import matplotlib.pyplot as plt
import glob
import os, errno
   
parser = argparse.ArgumentParser(description='ga')
parser.add_argument('--input', help='Name of input folder', default='./input')
parser.add_argument('--output', help='Name of output folder', default='./output')
args = parser.parse_args()
try:
    os.makedirs(args.output)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

image = cv.imread("{:s}//{:s}".format(args.input,"input_resized.png"),cv.IMREAD_UNCHANGED)

extracted = []
for extra in glob.glob("{:s}//{:s}".format(args.input,"c_*.png")):
    extracted.append(cv.imread(extra,cv.IMREAD_UNCHANGED))

(H, W) = image.shape[:2]

print("Score: identity={:.2f}, empty={:.2f}".format(ssim(image,image,multichannel=True),ssim(image,np.zeros(image.shape,dtype=np.uint8),multichannel=True)))

class Chunk:
    def __init__(self,img,scale,angle,pos):
        self.img = img
        self.scale = scale
        self.angle = angle
        self.pos = pos
    
    def rotate(self):
        #rotate_bound: img.shape->inf 
        #TODO square bounding box?
        self.img = imutils.rotate(self.img, self.angle)

    def resize(self):
        self.img = cv.resize(self.img,(int(self.img.shape[1]*self.scale[1]),int(self.img.shape[0]*self.scale[0])))

    def put(self):
        layer = np.zeros(image.shape, dtype=np.uint8)
        (h, w) = self.img.shape[:2]
        if self.pos[1] + h > H and self.pos[0] + w > W:
            layer[self.pos[1]:,self.pos[0]:] = self.img[:H-self.pos[1],:W-self.pos[0]]
            return layer
        elif self.pos[1] + h > H:
            layer[self.pos[1]:,self.pos[0]:self.pos[0]+w] = self.img[:H-self.pos[1],:]
            return layer
        elif self.pos[0] + w > W:
            layer[self.pos[1]:self.pos[1]+h,self.pos[0]:] = self.img[:,:W-self.pos[0]]
            return layer
        layer[self.pos[1]:self.pos[1]+h, self.pos[0]:self.pos[0]+w] = self.img
        return layer 

def draw(images):
    output = np.zeros(image.shape, dtype=np.uint8)
    for im in images:
        # im.rotate()
        # im.resize()
        layer = im.put()
        cnd = layer[:,:,3] > 0
        output[cnd] = layer[cnd]
    return output

def fitness_func(score):
    return score

def weighted_random_choice(fitness):
    max = sum(fitness.values())
    pick = np.random.uniform(0, max)
    current = 0
    for key, value in fitness.items():
        current += value
        if current > pick:
            return key

def get_best(generation,fitness,number_of_best):
    # TODO fix
    sorted_fitness = sorted(fitness, key=fitness.get)
    print(sorted_fitness[len(sorted_fitness) - number_of_best:])
    best = [generation[i] for i in sorted_fitness[len(sorted_fitness) - number_of_best:]]
    return best

def new_generation(parents, size):
    generation = []
    while len(generation) != size:
        # mutation only
        chunks = []
        len_imgs = len(parents)
        img1 = parents[np.random.randint(low=0,high=len_imgs)]
        for extra in img1:
            scale_x = np.random.normal(loc=extra.scale[1],scale=1) + 1
            scale_y = np.random.normal(loc=extra.scale[0],scale=1) + 1
            angle = np.random.normal(loc=extra.angle,scale=1)
            x = np.random.normal(loc=extra.pos[1],scale=1) 
            y = np.random.normal(loc=extra.pos[0],scale=1)
            chunks.append(Chunk(extra.img,[scale_y,scale_x],angle,[int(y),int(x)]))
        generation.append(chunks)
    return generation

def fitness(generation):
    fitness = {}
    for i in range(len(generation)):
        score = ssim(image,draw(generation[i]),multichannel=True)
        fitness[i] = fitness_func(score)
    return fitness

generation_size = 20
number_of_best = 2
generation = [[Chunk(img,[1,1],np.random.randint(low=-180,high=180),[np.random.randint(low=0,high=H),np.random.randint(low=0,high=W)]) for img in extracted] for i in range(generation_size)]

num_iterations = 10
iterations = np.arange(start=0,stop=num_iterations,step=1,dtype=np.uint8)
best_score = []
average_score = []
for i in iterations:
    print(i)
    fit = fitness(generation)
    output = draw(generation[max(fit.iteritems(), key=operator.itemgetter(0))[0]])
    cv.imwrite("{:s}/out_{:d}.png".format(args.output,i), output)   
    best_score.append(max(fit.iteritems(), key=operator.itemgetter(1))[1])
    average_score.append(sum(i for i in fit.values())/(len(fit.keys())))
    parents = [generation[weighted_random_choice(fit)] for i in range(int(generation_size/2) - number_of_best)]
    generation = new_generation(parents,generation_size)   
    # generation = generation + get_best(generation,fit,number_of_best)

plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
plt.plot(iterations,best_score,'-r',label="best")
plt.plot(iterations,average_score,'-b',label="average")
plt.legend()
plt.savefig("{:s}/plot.png".format(args.output))