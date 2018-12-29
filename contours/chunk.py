import numpy as np

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

    def put(self, shape):
        layer = np.zeros(shape, dtype=np.uint8)
      
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