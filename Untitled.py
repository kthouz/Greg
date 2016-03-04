
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from pprint import pprint
from scipy import spatial
from sklearn.preprocessing import normalize as Normalize
import cv2


# In[29]:

class GregImages():
    def __init__(self,directory,img_format='jpg'):
        self.path = directory
        self.img_filenames = glob(directory+'*.'+img_format)
    
    def load_img(self,img_filename,to_gray=False,display=False):
        try:
            img = cv2.imread(img_filename)
            if to_gray:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if display:
                plt.imshow(img)
                plt.show()
            return img
        except:
            print "image not found"
    
    def vectorize(self,img):
        import warnings
        warnings.filterwarnings('ignore')
        
        v = np.reshape(img,-1)
        return Normalize(v)
    
    def correlate_imgs(self,v1,v2):
        """
        calculate correlation between two images
        """
        if v1.shape == v2.shape:
            if (len(v1.shape)>1) or (len(v2.shape)>1):
                v1 = self.vectorize(v1)
                v2 = self.vectorize(v2)
            cor = 1-spatial.distance.cosine(v1,v2)
            return cor
        else:
            print "images should be of same dimension"
            return 0
        
def analyze_img(root,folder,strategy='first_frame'):
    main_dir = root
    folder = folder
    g = GregImages(main_dir+folder+'/')
    filenames = g.img_filenames[0:300]
    im0 = g.vectorize(g.load_img(filenames[0],display=False,to_gray=True))
    cor = []
    for i in range(1,len(filenames)):
        imi = g.vectorize(g.load_img(filenames[i],to_gray=True))
        cor.append(g.correlate_imgs(im0,imi))
        if strategy == 'previous_frame':
            im0 = g.vectorize(g.load_img(filenames[i],to_gray=True))
        else:
            #keep using the initial frame
            pass
    plt.plot(cor)
    plt.xlabel('frame #')
    plt.ylabel('correlation w.r.t 1st frame')
    plt.savefig(main_dir+folder+'.eps', format='eps',dpi=1000)
    plt.show()
    print 'figure saved as', main_dir+folder+'.eps'
    return cor


# In[30]:

if __name__ == '__main__':
    root = '/Users/cgirabawe/Documents/PhD/Collabo/Greg/img/'
    folder = '2'
    cor = analyze_img(root,folder,strategy='previous_frame')


# In[ ]:



