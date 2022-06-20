from tqdm.contrib.concurrent import process_map  # or thread_map
import multiprocessing as mp
import time
import cv2
import os
import glob
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import mahotas
from tqdm import tqdm
from six.moves import cPickle as pickle #for performance
from functools import partial


ena_local = 'C:/temp/ena/images/train100/'
images = [os.path.split(i)[1] for i in glob.glob(ena_local + '/*.jpg', recursive=True)]

def _getfeat(feats, image):
    img = cv2.imread(ena_local + image)
    b, g, r =  (mahotas.features.lbp(img[:, :, i], 2.5, 12) for i in (0, 1, 2))
    lpbrgb = np.stack((b,g,r))
    chists_reshaped = np.squeeze(np.stack(tuple(cv2.calcHist(img, [i], None, [352], [0,256]) for i in (0, 1, 2))))
    vstack = np.vstack((lpbrgb, chists_reshaped))
    feats[image] = vstack

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

if __name__ == '__main__':
    with mp.Manager() as manager:
        feats = manager.dict()
        process_map(partial(_getfeat, feats), images, max_workers=12, chunksize=2)
        save_dict(dict(feats), 'c:/temp/train100.pkl')

