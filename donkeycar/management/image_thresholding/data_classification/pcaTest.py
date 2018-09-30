# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:10:38 2018

@author: ejjackson
"""

import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import cross_validation
import matplotlib.pyplot as plt
#from sklearn.datasets import load_digits

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))


digits = datasets.load_digits()
#iris = datasets.load_iris()
X = digits.data
y = digits.target

y[y!=1]=0

print y
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
pca = PCA(n_components=3)# adjust yourself
pca.fit(X_train)
X_t_train = pca.transform(X_train)
X_t_test = pca.transform(X_test)
clf = SVC()
clf.fit(X_t_train, y_train)
filtered = pca.inverse_transform(X_t_train)
print 'score', clf.score(X_t_test, y_test)
#print 'actual label',y_test
#print 'pred label', clf.predict(X_t_test)