#!/usr/bin/env python3
"""
Look at a file and see if a mask exists for it. Analyse the image based on the mask

Usage:
    test.py (draw) [--fileName=<fn>] [--nCrop=<n1>] [--nThresh=<n1>] [--cd]
    test.py (folder) [--fileName=<fn>] [--nCrop=<n1>] [--nThresh=<n1>] [--cd] [--hsv]
    test.py (clf) [--fileName=<fn>] [--nCrop=<n1>] [--cd] [--hsv] [--s] [--test]

Options:
    -h --help        Show this screen.
"""
from docopt import docopt
import cv2
import numpy as np
import sys
import os.path
import matplotlib as mpl
import glob
from sklearn.svm import SVC
from sklearn import cross_validation
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fileStubs = ['_mask','_cones']

def analyseMasks(fileName,isCone,rgb = True,doPlot = True):
    if isCone == False:
        fileStub = '_mask'
    else:
        fileStub = '_cones'
    if '.jpg' in fileName:
        temp  = fileName.replace('.jpg','')
        fileName = temp
    if not os.path.isfile(fileName+'.jpg'):
        print('no file with that name')
        return
    elif os.path.isfile(fileName+fileStub+'.jpg') and os.path.isfile(fileName+fileStub+'.npy'):
        print('lets go')

        myMask = np.load(fileName+fileStub+'.npy')
        img1_orig = cv2.imread((fileName+'.jpg'))
        img1 = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB)
        img1_orig = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB)
        masked_data = cv2.bitwise_and(img1, img1, mask=myMask)
        masked_data_orig = cv2.bitwise_and(img1_orig, img1_orig, mask=myMask)
        unmasked_data_orig = cv2.bitwise_and(img1_orig, img1_orig, mask=1-myMask)
        masked_data2 = cv2.bitwise_and(img1, img1, mask=1-myMask)
        height,width,depth = img1.shape
        output1 = np.zeros((height,width), np.uint8)
        output2 = np.zeros((height,width), np.uint8)
        #cv2.inRange(img1, np.array([169, 10, 100]), np.array([183, 48, 145]), output1)
        #cv2.inRange(img1, np.array([0, 50, 100]), np.array([10, 160, 200]), output2)
        cv2.inRange(img1, np.array([169, 0, 0]), np.array([183, 255, 255]), output1)
        cv2.inRange(img1, np.array([0, 0, 0]), np.array([10, 255, 255]), output2)
        if doPlot == True:
            fig1 = plt.figure()
            ax1 = plt.subplot2grid((2,2),(0,0))
            ax2 = plt.subplot2grid((2,2),(0,1))
            ax3 = plt.subplot2grid((2,2),(1,0))
            ax4 = plt.subplot2grid((2,2),(1,1))
            ax1.imshow(img1_orig)
            ax2.imshow(myMask)
            ax3.imshow(masked_data_orig)



            fig2 = plt.figure()
            ax3d = fig2.add_subplot(221, projection='3d')
            ax5 = fig2.add_subplot(222)
            ax6= fig2.add_subplot(223)
            ax7= fig2.add_subplot(224)

            ax3d.scatter(masked_data2[0::10,0::10,0], masked_data2[0::10,0::10,1], masked_data2[0::10,0::10,2], c='b', marker='o')
            ax3d.scatter(masked_data[:,:,0], masked_data[:,:,1], masked_data[:,:,2], c='r', marker='o')
            ax5.scatter(masked_data2[:,:,0],masked_data2[:,:,1],c='b', marker='o')
            ax6.scatter(masked_data2[:,:,1],masked_data2[:,:,2],c='b', marker='o')
            ax7.scatter(masked_data2[:,:,0],masked_data2[:,:,2],c='b', marker='o')
            ax5.scatter(masked_data[0:,:,0],masked_data[0:,:,1],c='r', marker='o')
            ax6.scatter(masked_data[0:,:,1],masked_data[0:,:,2],c='r', marker='o')
            ax7.scatter(masked_data[0:,:,0],masked_data[0:,:,2],c='r', marker='o')
            ax5.set_xlabel('H')
            ax6.set_xlabel('s')
            ax7.set_xlabel('h')
            ax5.set_ylabel('s')
            ax6.set_ylabel('v')
            ax7.set_ylabel('v')
            ax3d.set_xlabel('H')
            ax3d.set_ylabel('S')
            ax3d.set_zlabel('V')
            ax4.imshow(output2)
            plt.show()
        return img1_orig,masked_data_orig,unmasked_data_orig




def analyseFolder(folderName = "testImages",isCone = True,doPlot = True,doHSV = True,nCrop = 0):
    all_orig = []
    masked_data = np.empty((0,3))
    unmasked_data = np.empty((0,3))
    for fn in glob.glob(folderName+'/'+'*.jpg'):
        if not any(stub in fn for stub in fileStubs):
            print(fn)
            o,m,u = analyseMasks(fn,isCone,doPlot = False)
            if doHSV:
                o = cv2.cvtColor(o, cv2.COLOR_RGB2HSV)
                m = cv2.cvtColor(m, cv2.COLOR_RGB2HSV)
                u = cv2.cvtColor(u, cv2.COLOR_RGB2HSV)
            if nCrop>0:
                o = o[nCrop:,:,:]
                m = m[nCrop:,:,:]
                u = u[nCrop:,:,:]

            m = m.reshape(-1,m.shape[-1])
            u = u.reshape(-1,u.shape[-1])
            m = m[~np.all(m==0,axis=1),:]
            u = u[~np.all(u==0,axis=1),:]
            masked_data = np.concatenate((masked_data,m),axis=0)
            unmasked_data = np.concatenate((unmasked_data,u),axis=0)
    print(masked_data.shape)
    print(unmasked_data.shape)
    if doPlot==True:
        fig1 = plt.figure()
        ax3d = fig1.add_subplot(221, projection='3d')
        if isCone==True:
            ustride = 1000
            mstride = 10
        else:
            ustride = 200
            mstride = 200
        myAlpha = 0.3
        ax3d.scatter(unmasked_data[0::ustride,0], unmasked_data[0::ustride,1], unmasked_data[0::ustride,2],
            c='b', marker='.', alpha = myAlpha)
        ax3d.scatter(masked_data[0::mstride,0], masked_data[0::mstride,1], masked_data[0::mstride,2],
            c='r', marker='.', alpha = myAlpha)
        ax3d.set_xlabel('R')
        ax3d.set_ylabel('G')
        ax3d.set_zlabel('B')
        ax5 = fig1.add_subplot(222)
        ax6 = fig1.add_subplot(223)
        ax7 = fig1.add_subplot(224)
        ax5.scatter(unmasked_data[0::ustride,0], unmasked_data[0::ustride,1],
            c='b', marker='.', alpha = myAlpha)
        ax5.scatter(masked_data[0::mstride,0], masked_data[0::mstride,1],
            c='r', marker='.', alpha = myAlpha)
        ax6.scatter(unmasked_data[0::ustride,0], unmasked_data[0::ustride,2],
            c='b', marker='.', alpha = myAlpha)
        ax6.scatter(masked_data[0::mstride,0],  masked_data[0::mstride,2],
            c='r', marker='.', alpha = myAlpha)
        ax7.scatter( unmasked_data[0::ustride,1], unmasked_data[0::ustride,2],
            c='b', marker='.', alpha = myAlpha)
        ax7.scatter( masked_data[0::mstride,1], masked_data[0::mstride,2],
            c='r', marker='.', alpha = myAlpha)
        plt.show()

    return masked_data,unmasked_data
    #img1_orig = cv2.imread((fileName+'.jpg'))

def doClf(isCone=True,doHSV = False,nCrop = 0,saveModel=False):
    #isCone=True
    masked_data,unmasked_data = analyseFolder(isCone=isCone,doHSV=doHSV,doPlot=False)

    if isCone==True:
        ustride = 400
        mstride = 2
    else:
        ustride = 50
        mstride = 50
    unmasked_data = unmasked_data[0::ustride,:]
    masked_data = masked_data[0::mstride,:]
    allData = np.concatenate([unmasked_data,masked_data],axis=0)
    print('allData.shape')
    print(allData.shape)
    labels = []
    for kk in range(len(unmasked_data)):
        labels.append(0)
    for kk in range(len(masked_data)):
        labels.append(1)
    print('length labels')
    print(len(labels))
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(allData, labels, test_size=0.4, random_state=1)
    print('X_train.shape')
    print(X_train.shape)
    clf = SVC(kernel = 'linear')
    clf.fit(X_train, y_train)
    labelPred = clf.predict(X_test)
    output = np.sum(np.equal(y_test,labelPred))/len(labelPred)
    pct_target = np.sum(y_test)/len(labelPred)
    print('percentage target class in test')
    print(pct_target)
    print('percentage correct')
    print(output)
    if saveModel:
        print('saving model')
        from sklearn.externals import joblib
        fileName = ''
        if isCone:
            fileName += 'cone_'
        else:
            fileName += 'track_'
        if doHSV:
            fileName += 'HSV_'
        else:
            fileName += 'RGB_'
        fileName += "%03d" % nCrop
        joblib.dump(clf, fileName+'.pkl')
    #clf.fit(scoreTrain, labelTrain)
def runClfImg(imgIn,clfIn):
    height,width,depth = imgIn.shape
    imgIn = imgIn.reshape(-1,imgIn.shape[-1])
    labelPred = clfIn.predict(imgIn)
    labelPred = labelPred.reshape(height,width)
    return labelPred
def testClfMask(folderName = "testImages",isCone = True,nCrop = 45):
    modelName = 'cone_RGB_000' if isCone else 'track_HSV_000'
    from sklearn.externals import joblib
    clf = joblib.load(modelName + '.pkl')
    for fn in glob.glob(folderName+'/'+'*.jpg'):
        if not any(stub in fn for stub in fileStubs):
            print(fn)
            o,m,u = analyseMasks(fn,isCone,doPlot = False)
            if nCrop>0:
                o = o[nCrop:,:,:]
                m = m[nCrop:,:,:]
                u = u[nCrop:,:,:]
            pred = runClfImg(o,clf)
            fig = plt.figure()
            ax1 = plt.subplot2grid((2,2),(0,0))
            ax2 = plt.subplot2grid((2,2),(0,1))
            ax3 = plt.subplot2grid((2,2),(1,0))
            ax4 = plt.subplot2grid((2,2),(1,1))
            ax1.imshow(o)
            ax2.imshow(m)
            ax3.imshow(u)
            ax4.imshow(pred)
    plt.show()





if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    cd = True if args['--cd'] else False
    hsv = True if args['--hsv'] else False
    if args['--nCrop']:
        nc = int(args['--nCrop'])
    else:
        nc = 0
    if args['--fileName']:
        analyseMasks(args['--fileName'],isCone = cd)
    if args['--fileName'] and args['--nCrop']:
        analyseMasks(args['--fileName'],int(args['--nCrop']),isCone = cd)
    if args['folder']:# and args['--folderName']:
        analyseFolder(isCone = cd, doHSV = hsv)
    if args['clf'] and args['--test']:
        testClfMask(isCone=cd,nCrop=nc)
    if args['clf'] and not args['--test']:# and args['--folderName']:
        sm = True if args['--s'] else False
        doClf(isCone = cd, doHSV = hsv,saveModel = sm)
