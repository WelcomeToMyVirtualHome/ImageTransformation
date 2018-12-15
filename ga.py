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

for img in glob.glob("{:s}//{:s}".format(args.output,"*.png")):
    os.remove(img)

image = cv.imread("{:s}//{:s}".format(args.input,"input_resized.png"),cv.IMREAD_UNCHANGED)

extracted = []
for img in glob.glob("{:s}//{:s}".format(args.input,"c_*.png")):
    extracted.append(cv.imread(img,cv.IMREAD_UNCHANGED))

(H, W) = image.shape[:2]
print('Image shape = [{:d},{:d}]'.format(H,W))
print("Score: identity={:.2f}, empty={:.2f}".format(ssim(image,image,multichannel=True),ssim(image,np.zeros(image.shape,dtype=np.uint8),multichannel=True)))

def bitfield(n):
    return [bool(digit) for digit in bin(n)[2:]]

def int_to_bitfield(integer,integer_max):
    if integer < integer_max:
        n = int(np.log2(integer_max) + 1)
        integer_bits = bitfield(integer)
        length = len(integer_bits)
        if n > length:
            integer_bits.reverse()
            integer_bits.extend(np.zeros(n - length,dtype=np.bool_))
            integer_bits.reverse()
        return integer_bits
    else:
        n = int(np.log2(integer) + 1)
        integer_bits = bitfield(integer)
        return integer_bits

def bitfield_to_int(bitfield):
    return sum([(2*int(bitfield[len(bitfield)-1-i]))**i for i in range(0,len(bitfield))])

class Chunk:
    def __init__(self,img,scale,angle,pos):
        self.img = img
        self.scale = scale
        self.angle = angle
        self.pos = pos
    
    def rotate(self):
        #TODO fix
        self.img = imutils.rotate(self.img, self.angle)

    def resize(self):
        # can't rescale to shape=(0,0)
        self.img = cv.resize(self.img,(int(self.img.shape[1]*self.scale[1]),int(self.img.shape[0]*self.scale[0])))

    def put(self):
        layer = np.zeros(image.shape, dtype=np.uint8)
      
        x, y = self.pos[1], self.pos[1]
        h1, w1 = layer.shape[:2]
        h2, w2 = self.img.shape[:2]

        x1min = max(0, x)
        y1min = max(0, y)
        x1max = max(min(x + w2, w1), 0)
        y1max = max(min(y + h2, h1), 0)

        x2min = max(0, -x)
        y2min = max(0, -y)
        x2max = min(-x + w1, w2)
        y2max = min(-y + h1, h2)

        layer[y1min:y1max, x1min:x1max] += self.img[y2min:y2max, x2min:x2max]
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
    generation = []
    mutation_p = 0.05
    while True:
        chunks = []
        len_imgs = len(parents)
        ind1 = np.random.randint(low=0,high=len_imgs)
        ind2 = np.random.randint(low=0,high=len_imgs)
        while ind1 == ind2:
            ind2 = np.random.randint(low=0,high=len_imgs)
        img1 = parents[ind1]
        img2 = parents[ind2]

        for extra1,extra2 in zip(img1,img2):
            pos_1y, pos_1x = int_to_bitfield(extra1.pos[0],H), int_to_bitfield(extra1.pos[1],W)
            pos_2y, pos_2x = int_to_bitfield(extra2.pos[0],H), int_to_bitfield(extra2.pos[1],W)

            cross = np.random.randint(low=1,high=len(pos_1y)-1)
            pos_chy = pos_1y[:cross] + pos_2y[cross:]
            
            cross = np.random.randint(low=1,high=len(pos_1x)-1)
            pos_chx = pos_1x[:cross] + pos_2x[cross:]
            
            mutation_ps = np.random.random(len(pos_chx)+len(pos_chy))
            for i,(x,y) in enumerate(zip(pos_chx,pos_chy)):
                if mutation_ps[i] < mutation_p:
                    pos_chy[i] = not y 
                if mutation_ps[i+len(pos_chy)] < mutation_p:
                    pos_chx[i] = not x
                
            
            scale_y = np.abs(np.random.normal(loc=extra1.scale[0],scale=1))
            scale_x = np.abs(np.random.normal(loc=extra1.scale[1],scale=1))
            angle = np.random.normal(loc=extra1.angle,scale=1)
            
            chunks.append(Chunk(extra1.img,[scale_y,scale_x],angle,[bitfield_to_int(pos_chy)%H,bitfield_to_int(pos_chx)%W]))
        generation.append(chunks)
        if len(generation) == size - number_of_best:
            break
    return generation

def fitness_func(score,i):
    # return some score function
    return score

def fitness(generation, i):
    n_fitness = {}
    fitness = {}
    for g in range(len(generation)):
        score = ssim(image,draw(generation[g]),multichannel=True)
        n_fitness[g] = fitness_func(score,i)
        fitness[g] = score
    return fitness, n_fitness

generation_size = 10
number_of_best = 2
generation = [[Chunk(img,[1,1],np.random.randint(low=-180,high=180),[np.random.randint(low=0,high=H),np.random.randint(low=0,high=W)]) for img in extracted] for i in range(generation_size)]

num_iterations = 100
iterations = np.arange(start=0,stop=num_iterations,step=1)
best_score = []
average_score = []
for i in iterations:
    print("i={:d}".format(i))
    (fit, n_fit) = fitness(generation,i)
    output = draw(generation[max(fit.iteritems(), key=operator.itemgetter(0))[0]])
    if i % 10 == 0:
        cv.imwrite("{:s}/out_{:d}.png".format(args.output,i), output)   
    best_score.append(max(fit.iteritems(), key=operator.itemgetter(1))[1])
    average_score.append(sum(fit.values())/(len(fit.values())))
    parents = [generation[weighted_random_choice(n_fit)] for i in range(int(generation_size/2) - number_of_best)]
    n_generation = new_generation(parents,generation_size,number_of_best)   
    generation = n_generation + get_best(generation,n_fit,number_of_best)

plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
plt.plot(iterations,best_score,'-r',label="best")
plt.plot(iterations,average_score,'-b',label="average")
plt.legend()
plt.savefig("{:s}/plot.png".format(args.output))