import numpy as np
import cv2

def load_data(size):


    with open('./assets/dataset.csv','r') as f:
        for line in f:
            l = line.split(',')
            path = l[0]
            if(path == 'filename'): continue
            # label = l[1].replace('\n','')
            img = cv2.imread(path)
            dim = (size, size)
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(path, resized)


load_data(150)