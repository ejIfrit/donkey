# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 16:54:52 2018

@author: ejjackson
"""

import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import cross_validation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pickle

#favorite_color = pickle.load( open( "save.p", "rb" ) )
pcas = pickle.load(open( "pcas.p", "rb" ) )
pca3 = pickle.load(  open( "pcaAll.p", "rb" ) )
clfScores = pickle.load(  open( "clfScores.p", "rb" ) )
clfAll =  pickle.load( open( "clfAll.p", "rb" ) )
size_orig = (32,45)
nComps = 3
fig = plt.figure()
for i in range(nComps): 
    ax = fig.add_subplot(5, 5, i+1+5, xticks=[], yticks=[]) 
    ax.imshow(np.reshape(pcas[1].components_[i,:], size_orig))

plt.show()

sys.stdout.flush()