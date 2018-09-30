#!/usr/bin/env python3
"""
look at an image file and try to extract the coordinates of the white dots

Usage:
    test.py (find) [--fileName=<fn>] [--nCrop=<n1>] [--nThresh=<n1>]

Options:
    -h --help        Show this screen.
"""
from docopt import docopt
import cv2
import numpy as np
import sys
import os.path
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def findLineMarkersFile(fileName,nCrop = 45,nThresh= 100,doPlot = True):
    if not os.path.isfile(fileName+'.jpg'):
        print('no file with that name')
        return
    else:
        img1 = cv2.imread((fileName+'.jpg'))
    findLineMarkers(img1,nCrop,nThresh,doPlot)

def findLineMarkers(img1,nCrop = 45,nThresh= 100,doPlot = True):
    # load image

    if doPlot:
        fig1 = plt.figure()
        ax1 = plt.subplot2grid((3,1),(0,0))
        ax2 = plt.subplot2grid((3,1),(1,0))
        ax3 = plt.subplot2grid((3,1),(2,0))
    # step 1- crop image- there are no road markers in the ceiling
    img1_crop = img1[nCrop:,:,:]
    height,width,depth = img1_crop.shape
    img1_thresh = np.zeros((height,width), np.uint8)


    # step 2- threshold the image to only find white pixels
    cv2.inRange(img1_crop.astype('uint8'), np.array([nThresh, nThresh, nThresh]), np.array([255, 255, 255]), img1_thresh)
    # grab contours
    _,cs,_ = cv2.findContours( img1_thresh.astype('uint8'), mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE )

    cv2.drawContours(img1_crop, cs, -1, (0,255,0), thickness=1)
    filledI = np.zeros(img1_thresh.shape[0:2]).astype('uint8')
    cv2.drawContours(filledI, cs, -1, 255, -1)
    allA = []
    filledA = []
    allXY = [[],[]]
    filledXY = [[],[]]
    nContours = []
    i = 0
    for c in cs:
        temp = np.zeros(img1_thresh.shape[0:2]).astype('uint8')
        m = cv2.moments(c)
        area          = m['m00']
        if area>0 and area<500:
            centroid      = ( m['m10']/m['m00'],m['m01']/m['m00'] )

            totalA = np.sum(cv2.drawContours(temp, cs, i, 255, thickness=-1))
            andA = np.sum(cv2.bitwise_and(filledI, temp))

            if totalA==andA:
                filledA.append(area)
                filledXY[0].append(centroid[0])
                filledXY[1].append(centroid[1])
                nContours.append(i)
            else:
                allA.append(area)
                allXY[0].append(centroid[0])
                allXY[1].append(centroid[1])

        i+=1



    if doPlot:
        ax3.imshow(img1_crop)
        ax3.plot(allXY[0],allXY[1],linestyle = 'None',marker = '.')
        ax3.plot(filledXY[0],filledXY[1],linestyle = 'None',marker = '.')

        for i, txt in enumerate(nContours):
            ax3.annotate(txt, (filledXY[0][i], filledXY[1][i]))


    print(len(cs))
    print(len(allA))
    print(len(filledA))
    sys.stdout.flush()
    if doPlot:
        cv2.imshow('cropped',img1_crop)
        cv2.imshow('thresholded',img1_thresh)
        cv2.imshow('filled',filledI)
    temp = np.zeros(img1_thresh.shape[0:2]).astype('uint8')
    #nContour = 13
    #cv2.imshow('contour1',cv2.drawContours(temp, cs, nContour, 255, -1))
    #cv2.imshow('anded',cv2.bitwise_and(filledI,cv2.drawContours(temp, cs, nContour, 255, -1)))
    #cv2.imshow('contour',imageContour)
    if doPlot:
        ax1.plot(allXY[0],allA,linestyle = 'None',marker = '.')
        ax2.plot(allXY[1],allA,linestyle = 'None',marker = '.')
        ax1.plot(filledXY[0],filledA,linestyle = 'None',marker = '.')
        ax2.plot(filledXY[1],filledA,linestyle = 'None',marker = '.')
        plt.show()
    #cv2.waitKey(0)
    return filledXY, filledA, allXY, allA

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)

    if args['--fileName'] and args['--nCrop'] and args['--nThresh']:
        findLineMarkersFile(args['--fileName'],int(args['--nCrop']),int(args['--nCrop']))
    elif args['--fileName']  and args['--nThresh']:
        findLineMarkersFile(args['--fileName'],nThresh = int(args['--nThresh']))
    elif args['--fileName']:
        findLineMarkersFile(args['--fileName'])
    else:
        print('you have to specify at least a filename')
