import glob
import cv2
import numpy as np

class BatchGenerator:
    def __init__(self, img_size, datadir, aug=True):
        self.folderPath = datadir
        self.imagePath = glob.glob(self.folderPath+"/*")
        self.datalen = len(self.imagePath)
        self.imgSize = (img_size,img_size)
        self.aug = aug

    def augment(self,img):
        rand = np.random.rand()
        if rand > .5:
            img = cv2.flip(img,1)
        return img

    def getBatch(self,nBatch,alpha=1.0):
        id = np.random.choice(range(self.datalen),nBatch)
        x   = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],3), dtype=np.float32)
        for i,j in enumerate(id):
            img = cv2.imread(self.imagePath[j])
            if self.aug :
                img = self.augment(img)
            img_x = cv2.resize(img,self.imgSize)
            if alpha < 1:
                img_y = cv2.resize(img,(self.imgSize[0]//2,self.imgSize[1]//2), interpolation=cv2.INTER_LINEAR)
                img_y = cv2.resize(img_y,(self.imgSize[0],self.imgSize[1]), interpolation=cv2.INTER_LINEAR)
                img_x = img_x * alpha + img_y * (1-alpha)
            x[i,:,:,:] = (img_x - 127.5) / 127.5
        return x
