# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:10:38 2018

@author: ejjackson
"""
import sys
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
#from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn import cross_validation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler

#from sklearn.datasets import load_digits

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))

def plot_specs(data):
    fig, axes = plt.subplots(4, 5, figsize=(8, 5),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        if i<len(data):
            ax.imshow(data[i])

#
digits = datasets.load_digits()
#iris = datasets.load_iris()

#print digits.target
noTrip = glob.glob('Z:\Project Share Folder\Ed_Jackson\UWB\spectrograms\MatSlow0*.npy')
yesTrip = glob.glob('Z:\Project Share Folder\Ed_Jackson\UWB\spectrograms\MatSlow1*.npy')

data = []
labels = []
data2 = []
labels2 = []
dataOrig = []
dataOrig2 = []

for kk in range(len(noTrip)):
    temp = np.load(noTrip[kk])
    iMax = np.argmax(temp.sum(axis=0))
    print iMax
    temp = np.roll(temp,22-iMax , axis=1)
    data.append(temp.flatten())
    dataOrig.append(temp)
    labels.append(0)

for kk in range(len(yesTrip)):
    temp = np.load(yesTrip[kk])

    iMax = np.argmax(temp.sum(axis=0))
    print iMax
    temp = np.roll(temp,22-iMax , axis=1)
    data2.append(temp.flatten())
    dataOrig2.append(temp)
    labels2.append(1)

size_orig = dataOrig[0].shape

plot_specs(dataOrig)
plot_specs(dataOrig2)

z_scaler = StandardScaler()
#data = z_scaler.fit_transform(data)
#data2 = z_scaler.fit_transform(data2)
X = np.array(data)
y = labels
X2 = np.array(data2)
y2 = labels2

#y[y!=1]=0
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=1)
X2_train, X2_test, y2_train, y2_test = cross_validation.train_test_split(X2, y2, test_size=0.4, random_state=1)

nComps = 3
pca = PCA(n_components=nComps)# adjust yourself
pca2 = PCA(n_components=nComps)# adjust yourself
pca.fit(np.array(X_train))
pca2.fit(np.array(X2_train))

#kpca = KernelPCA(n_components=3)
#kpca_transform = pca.fit_transform(X_train)
fig, ax1, = plt.subplots(1,1)
plt.semilogy(pca.explained_variance_ratio_, '--o')
plt.semilogy(pca.explained_variance_ratio_.cumsum(), '--o');
plt.semilogy(pca2.explained_variance_ratio_, '--o')
plt.semilogy(pca2.explained_variance_ratio_.cumsum(), '--o');
X_t_train = pca.transform(X_train)
test1 = pca.score_samples(X_test)
test2 = pca2.score_samples(X_test)
test3 = pca.score_samples(X2_test)
test4 = pca2.score_samples(X2_test)

print test1.shape

fig = plt.figure()
for i in range(nComps):
    ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
    ax.imshow(np.reshape(pca.components_[i,:], size_orig))
for i in range(nComps):
    ax = fig.add_subplot(5, 5, i+11, xticks=[], yticks=[])
    ax.imshow(np.reshape(pca2.components_[i,:], size_orig))


print pca.components_.shape
print pca.singular_values_


train1 = pca.score_samples(X_train)
train2 = pca.score_samples(X2_train)
train3 = pca2.score_samples(X_train)
train4 = pca2.score_samples(X2_train)
scoreTrain = np.transpose(np.stack([np.concatenate([train1,train2]),np.concatenate([train3,train4])]))
labelTrain = np.concatenate([y_train,y2_train])
print scoreTrain
print scoreTrain.shape
print labelTrain

data = (np.stack([train1,train3]),np.stack([train2,train4]))
colors = ("C0", "C1")
groups = ("Type 1", "Type 2")


fig = plt.figure()
for i in range(len(X_test)):
    ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
    ax.imshow(np.reshape(X_test[i,:], size_orig))
for i in range(len(X2_test)):
    ax = fig.add_subplot(5, 5, i+11, xticks=[], yticks=[])
    ax.imshow(np.reshape(X2_test[i,:], size_orig))

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

for data, color, group in zip(data, colors, groups):
    x, y = data
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)


clf = SVC(kernel = 'linear')
clf.fit(scoreTrain, labelTrain)
test1 = pca.score_samples(X_test)
test2 = pca.score_samples(X2_test)
test3 = pca2.score_samples(X_test)
test4 = pca2.score_samples(X2_test)
scoreTest = np.transpose(np.stack([np.concatenate([test1,test2]),np.concatenate([test3,test4])]))
labelPred = clf.predict(scoreTest


print '-----------'
print labelPred
print np.concatenate([y_test,y2_test])


data = (np.stack([test1,test3]),np.stack([test2,test4]))
colors = ("C0", "C1")
groups = ("Type 1", "Type 2") 

for data, color, group in zip(data, colors, groups):
    x, y = data
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group,marker = '+')

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5120, -5000)
yy = a * xx - (clf.intercept_[0]) / w[1]

ax.plot(xx, yy, 'k-')

nComps = 2
pca3 = PCA(n_components=nComps)# adjust yourself

pca3.fit(np.concatenate([X_train,X2_train]))
trainResult = pca3.transform(np.concatenate([X_train,X2_train]))
testResult = pca3.transform(np.concatenate([X_test,X2_test]))


fig = plt.figure()
for i in range(nComps):
    ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
    ax.imshow(np.reshape(pca3.components_[i,:], size_orig))

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
for c, m, temp in [('C0', 'o',X_train), ('C1', 'o',X2_train),('C0', '+',X_test), ('C1', '+',X2_test)]:
    temp = pca3.transform(np.concatenate([temp]))
    xs = temp[:,0]
    ys = temp[:,1]
    #zs = temp[:,2]
    #ax.scatter(xs, ys, zs,c=c,marker=m)
    ax.scatter(xs, ys,c=c,marker=m)

clf2 = SVC(kernel = 'linear')
clf2.fit(trainResult, labelTrain)
testLabels = clf2.predict(testResult)

w = clf2.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-50, 100)
yy = a * xx - (clf2.intercept_[0]) / w[1]

ax.plot(xx, yy, 'k-')



print testLabels
sys.stdout.flush()

#data = np.concatenate(data,data2,axis=0)
#labels = np.concatenate(labels,labels2,axis=0)
#
#print data.shape
#print labels.shape

##filtered = pca.inverse_transform(X_t_train)
#print 'score', clf.score(X_t_test, y_test)
#print 'actual label',y_test
#print 'pred label', clf.predict(X_t_test)
#X_t = pca.transform(X)
#
#print 'score', clf.score(X_t, y)
#print 'actual label',y
#print 'pred label', clf.predict(X_t)
plt.show()
