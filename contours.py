from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse
import random as rng
import imutils
from skimage.measure import compare_ssim as ssim
   
parser = argparse.ArgumentParser(description='Contours')
parser.add_argument('--input', help='Path to input image.png', default='shapes.png')
args = parser.parse_args()
src = cv.imread(args.input)
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

image = cv.imread(args.input,cv.IMREAD_UNCHANGED)

hight = 500
(h,w) = image.shape[:2]
scale = float(hight)/h
n_h,n_w = hight, image.shape[0]*scale
image = cv.resize(image,(int(n_h),int(n_w)))

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
# dilated = cv.dilate(gray, kernel)
closing = cv.GaussianBlur(closing, (3, 3), 0)
edged = cv.Canny(closing, 50, 100)
cv.imshow("image",edged)
cv.waitKey(0)
_, cnts, _ = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
(H, W) = image.shape[:2]
    
def image_in_contour(image, contour):
    (x, y, w, h) = cv.boundingRect(contour)
    transparent_copy = np.zeros(image.shape,dtype="uint8")
    copy = image.copy()
    cv.drawContours(copy, [contour], -1, (255,255,255,255), 0)
    for xx in range(copy.shape[1]):
        for yy in range(copy.shape[0]):
            if cv.pointPolygonTest(contour,(xx,yy),False) > 0:
                transparent_copy[yy,xx] = copy[yy,xx]
    return transparent_copy[y:y + h, x:x + w] 

class Chunk:
    def __init__(self,img,scale,angle,pos):
        self.img = img
        self.scale = scale
        self.angle = angle
        self.pos = pos
    
    def rotate(self):
        self.img = imutils.rotate_bound(self.img, self.angle)

    def resize(self):
        self.img =  cv.resize(self.img,(int(self.img.shape[1]*self.scale[1]),int(self.img.shape[0]*self.scale[0])))

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

extracted = []
for c in cnts:
    extracted.append(image_in_contour(image,c))

output = np.zeros(image.shape, dtype=np.uint8)
images = [Chunk(img,[1,1],0,[0,0]) for img in extracted]

for im in images:
    im.pos[0] = np.random.randint(low=0,high=image.shape[0])
    im.pos[1] = np.random.randint(low=0,high=image.shape[1])
    im.angle = np.random.randint(low=-180,high=180)
    im.scale[0] = np.random.rand()*2
    im.scale[1] = np.random.rand()*2

    im.rotate()
    im.resize()
    layer = im.put()
    cnd = layer[:,:,3] > 0
    output[cnd] = layer[cnd]

cv.imwrite(args.input[:args.input.find(".png")  ]+"_mix.png", output)   

score = ssim(image,output,multichannel=True)
print("SSIM = {:f}".format(score))

##### TODO:

def fitness():
    pass

def get_best():
    pass

def new_generation():
    pass

def loop():
    pass