# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:10:38 2018

@author: ejjackson
"""
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import svm
from sklearn import cross_validation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.font_manager
import glob
import pickle

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
            temp = np.abs(data[i])
            temp = data[i].sum(axis=0)
            #temp = np.cumsum(temp)
            temp = (temp-min(temp))/(max(temp)-min(temp))*32
           
            temp2 = np.cumsum(np.array(temp)-min(temp))       
            
            temp2 = (temp2-min(temp2))/(max(temp2)-min(temp2))
            midPoint = np.amin(np.where(temp2 > 0.5))
            print midPoint
            sys.stdout.flush()       
            ax.plot(np.array(temp2*32))
            ax.plot([midPoint,midPoint],[0,32])

def myRoll(data,doMax = True):
            temp = data.sum(axis=0)
            temp = (temp-min(temp))/(max(temp)-min(temp))*32
            if doMax == True:
                midPoint = np.argmax(temp)
            else:
                
                temp2 = np.cumsum(np.array(temp)-min(temp))       
                temp2 = (temp2-min(temp2))/(max(temp2)-min(temp2))
                midPoint = np.amin(np.where(temp2 > 0.5))
            temp = np.roll(data,22-midPoint , axis=1)
            return temp

def preProc(fileNamesIn,labelNo,doMax = True):
    data = []
    labels = []
    dataOrig = []
    for kk in range(len(fileNamesIn)):
        temp = np.load(fileNamesIn[kk])
        temp2 = myRoll(temp,doMax)
        data.append(temp2.flatten())
        dataOrig.append(temp2)
        labels.append(labelNo)
    plot_specs(dataOrig)
    size_orig = dataOrig[0].shape
    return data,labels,size_orig

def getPCA(data,labels,nComps=3):
    X = np.array(data)
    y = labels    
    #y[y!=1]=0
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=1)    
    pca = PCA(n_components=nComps)# adjust yourself
    pca.fit(np.array(X_train))
    return pca, X_train, X_test, y_train, y_test
if __name__ == "__main__":
    labelCnt = [0,1,2]
    nComps = 3
    pcas = []
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for labelNo in labelCnt:
        fileNames = glob.glob('Z:\Project Share Folder\Ed_Jackson\UWB\spectrograms\MatSlow'+str(labelNo)+'*.npy')
        sys.stdout.flush()
        data,labels,size_orig = preProc(fileNames,labelNo,False)
        pcaOut, X_trainOut, X_testOut, y_trainOut, y_testOut = getPCA(data,labels,nComps)
        pcas.append(pcaOut)
        X_train.append(X_trainOut)
        X_test.append(X_testOut)
        y_train.append(y_trainOut)
        y_test.append(y_testOut)
        
    print 'size_orig'    
    print size_orig
    allScoresIn = np.array([])
    allLabelsIn = np.array([])
    allScoresOut = np.array([])
    allLabelsOut = np.array([])
    allX_train = np.array([])
    allX_test = np.array([])
    cnt=0
    fig = plt.figure()
    for pca in pcas:
        for i in range(nComps): 
            ax = fig.add_subplot(5, 5, i+1+5*cnt, xticks=[], yticks=[]) 
            
            ax.imshow(np.reshape(pca.components_[i,:], size_orig))
        tempScores = np.array([])
        tempScoresOut = np.array([])
        for x, y, xt, yt in zip(X_train, y_train, X_test, y_test):
            scoresIn = pca.score_samples(x)
            scoresOut = pca.score_samples(xt)
            tempScores = np.concatenate([tempScores,scoresIn])
            tempScoresOut = np.concatenate([tempScoresOut,scoresOut])
            
            if cnt==0:
                allLabelsIn = np.concatenate([allLabelsIn,y])
                allLabelsOut = np.concatenate([allLabelsOut,yt])
                print '-----allX_train-------' 
                print allX_train.shape
                print x.shape
                sys.stdout.flush()
                if allX_train.size:
                    allX_train = np.vstack([allX_train,x])
                    allX_test = np.vstack([allX_test,xt])
                else:
                    allX_train = x
                    allX_test = xt
        
        cnt+=1
        print '------------'    
        print allScoresIn.shape
        print tempScores.shape
        if allScoresIn.size: 
            allScoresIn = np.vstack([allScoresIn,tempScores])
            allScoresOut = np.vstack([allScoresOut,tempScoresOut])
        else: 
            allScoresIn = tempScores
            allScoresOut = tempScoresOut
    
    pca3 = PCA(n_components=nComps)
    pca3.fit(allX_train)
    
    for i in range(nComps): 
        ax = fig.add_subplot(5, 5, i+1+5*cnt, xticks=[], yticks=[]) 
        ax.imshow(np.reshape(pca3.components_[i,:], size_orig))
    print '----sizes------'
    print pca3.transform(allX_train).shape
    print allScoresIn.shape
    print allLabelsIn.shape
    sys.stdout.flush()
    clfScores = SVC(kernel = 'linear')
    clfScores.fit(np.transpose(allScoresIn),allLabelsIn)
    testLabelsScores = clfScores.predict(np.transpose(allScoresOut))
    #clf2.fit(trainResult, labelTrain)
    #testLabels = clf2.predict(testResult)
    
    clfAll = SVC(kernel = 'linear')
    clfAll.fit(pca3.transform(allX_train), allLabelsIn)
    testLabels = clfAll.predict(pca3.transform(allX_test))
    print '------RESULTS-------'
    print allLabelsOut
    print testLabels
    print testLabelsScores
    sys.stdout.flush()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.add_subplot(111)
    for c, m, temp in [('C0', 'o',X_train[0]), ('C1', 'o',X_train[1]),('C2', 'o',X_train[2]), ('C0', '+',X_test[0]),('C1', '+',X_test[1]), ('C2', '+',X_test[2])]:
        temp = pca3.transform(np.concatenate([temp]))
        xs = temp[:,0]
        ys = temp[:,1]
        zs = temp[:,2]
        ax.scatter(xs, ys, zs,c=c,marker=m)
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.add_subplot(111)
    for c, m, temp in [('C0', 'o',X_train[0]), ('C1', 'o',X_train[1]),('C2', 'o',X_train[2]), ('C0', '+',X_test[0]),('C1', '+',X_test[1]), ('C2', '+',X_test[2])]:
        #temp = pca3.transform(np.concatenate([temp]))
        xs = pcas[0].score_samples(temp)
        ys = pcas[1].score_samples(temp)
        zs = pcas[2].score_samples(temp)
        ax.scatter(xs, ys, zs,c=c,marker=m)
    
    if True:
        pickle.dump( pcas, open( "pcas2.p", "wb" ) )
        pickle.dump( pca3, open( "pcaAll2.p", "wb" ) )
        pickle.dump( clfScores, open( "clfScores2.p", "wb" ) )
        pickle.dump( clfAll, open( "clfAll2.p", "wb" ) )
    
    
    #
    
    print 'minScore1 = ',min(pcas[0].score_samples(X_train[1])),min(pcas[0].score_samples(X_test[1]))
    print 'minScore2 = ',min(pcas[1].score_samples(X_train[1])),min(pcas[1].score_samples(X_test[1]))
    print 'minScore3 = ',min(pcas[2].score_samples(X_train[1])),min(pcas[2].score_samples(X_test[1]))
    print 'maxScore1 = ',max(pcas[0].score_samples(X_train[1])),max(pcas[0].score_samples(X_test[1]))
    print 'maxScore2 = ',max(pcas[1].score_samples(X_train[1])),max(pcas[1].score_samples(X_test[1]))
    print 'maxScore3 = ',max(pcas[2].score_samples(X_train[1])),max(pcas[2].score_samples(X_test[1]))
    
    
    xs = (pcas[0].score_samples(X_train[1])+5200.)/200.
    ys = (pcas[1].score_samples(X_train[1])+5300.)/200.
    xs2 =(pcas[0].score_samples(X_train[0])+5200.)/200.
    ys2 = (pcas[1].score_samples(X_train[0])+5300.)/200.
    xs3 = (pcas[0].score_samples(X_train[2])+5200.)/200.
    ys3 = (pcas[1].score_samples(X_train[2])+5300.)/200.
    xtest = (pcas[0].score_samples(X_test[1])+5200.)/200.
    ytest = (pcas[1].score_samples(X_test[1])+5300.)/200.
    scorestest = np.stack([xtest,ytest])
    
    print '-------1D clf--------'
    print xs.shape
    print ys.shape
    scores2d = np.stack([xs,ys])
    notscores = np.concatenate([np.stack([xs2,ys2]),np.stack([xs3,ys3])],axis=1)
    print scores2d.shape
    print scorestest.shape
    print notscores.shape
    sys.stdout.flush()
    clf1class = svm.OneClassSVM(nu=0.1, kernel="rbf")
    clf1class.fit(np.transpose(scores2d))
    y_pred_train = clf1class.predict(np.transpose(scores2d))
    y_pred_test = clf1class.predict(np.transpose(scorestest))
    y_pred_outliers = clf1class.predict(np.transpose(notscores))
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
    
    xx, yy = np.meshgrid(np.linspace(min(np.concatenate((xs,xs2,xs3))),max(np.concatenate((xs,xs2,xs3))) , 500), np.linspace(min(np.concatenate((ys,ys2,ys3))),max(np.concatenate((ys,ys2,ys3))), 500))
                                      
                                      
    # plot the line, the points, and the nearest vectors to the plane
    Z = clf1class.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.title("Novelty Detection")
    print '-----zmin zmax-----------'
    print Z.min(), Z.max()
    sys.stdout.flush()
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 7), cmap=plt.cm.PuBu)
    #a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    #plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
    #
    s = 40
    b1 = plt.scatter(xs, ys, c='white', s=s, edgecolors='k')
    b2 = plt.scatter(xtest, ytest, c='blueviolet', s=s,
                     edgecolors='k')
    c = plt.scatter(np.concatenate((xs2,xs3)), np.concatenate((ys2,ys3)), c='gold', s=s,
                    edgecolors='k')
    plt.axis('tight')
    #plt.xlim((-5, 5))
    #plt.ylim((-5, 5))
    #plt.legend([a.collections[0], b1, b2, c],
    #           ["learned frontier", "training observations",
    #            "new regular observations", "new abnormal observations"],
    #           loc="upper left",
    #           prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        "error train: %d/200 ; errors novel regular: %d/40 ; "
        "errors novel abnormal: %d/40"
        % (n_error_train, n_error_test, n_error_outliers))
    
    plt.show()