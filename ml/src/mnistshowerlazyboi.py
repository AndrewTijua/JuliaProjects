# -*- coding: utf-8 -*-
"""
Created on Tue May 19 00:36:19 2020

@author: Ben
"""

import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt(open("F:/julia/projects/ml/src/data/optdigitstes.csv", "rb"), delimiter=",", skiprows=1)

img = data[:,0:64]

imgind = 779

tpimg = np.array(img[imgind])
tpimg = np.reshape(tpimg, (8, 8))
plt.imshow(tpimg, cmap = 'Greys')

tpimg_long = np.reshape(tpimg, (1, 64))
#plt.imshow(tpimg_long, cmap = 'Greys')