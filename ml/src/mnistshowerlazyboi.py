# -*- coding: utf-8 -*-
"""
Created on Tue May 19 00:36:19 2020

@author: Ben
"""

import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt(open("F:/julia/projects/ml/data/opdig_lrg_tr.csv", "rb"), delimiter=",", skiprows=1)

#img = data[:,0:64]
img = data[:,:]

imgind = 1

tpimg = np.array(img[imgind])
# tpimg = np.reshape(tpimg, (8, 8))
tpimg = np.reshape(tpimg, (9, 9))
plt.imshow(tpimg, cmap = 'Greys')
