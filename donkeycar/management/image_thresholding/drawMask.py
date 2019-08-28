#!/usr/bin/env python3
"""
Look at a file and see if a mask exists for it. If not, draw it.

Usage:
    test.py (draw) [--fileName=<fn>] [--nCrop=<n1>] [--nThresh=<n1>] [--coneDraw] [--lineDraw] [--ylineDraw]

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
from mpl_toolkits.mplot3d import Axes3D
def drawMask(fileName = 'testImg2',nCrop = 0, coneDraw = False, lineDraw = False, ylineDraw = False):
    drawing = False
    maskDraw = False
    if coneDraw == False and lineDraw == False and ylineDraw == False:
        maskDraw = True
    if coneDraw == False:
        fileStub = '_mask'
    else:
        fileStub = '_cones'
    if lineDraw:
        fileStub = '_line'
    if ylineDraw:
        fileStub = '_yline'
    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        global drawing,mode
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            #ix,iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.circle(img2,(x,y),2,(255,0,0),-1)
                cv2.circle(myMask,(x,y),2,1,-1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img2,(x,y),1,(255,0,0),-1)
            cv2.circle(myMask,(x,y),1,-1)
    def draw_circleSmall(event,x,y,flags,param):
        global drawing,mode
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            #ix,iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.circle(img2,(x,y),1,(255,69,0),-1)
                cv2.circle(myMask,(x,y),1,1,-1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img2,(x,y),1,(255,69,0),-1)
            cv2.circle(myMask,(x,y),1,-1)

    if not os.path.isfile(fileName+'.jpg'):
        print('no file with that name')
        return
    elif os.path.isfile(fileName+fileStub+'.jpg') and os.path.isfile(fileName+fileStub+'.npy'):
        fig1 = plt.figure()
        myMask = np.load(fileName+fileStub+'.npy')
        plt.imshow(myMask)

        img1 = cv2.imread((fileName+'.jpg'))
        img2 = cv2.imread((fileName+fileStub+'.jpg'))
        dst = img1.copy()
        alpha = 0.4
        cv2.addWeighted( img2, alpha, img1, 1.-alpha,0, dst);
        cv2.imshow('image',dst)
        plt.show()
    else:
        img1 = cv2.imread((fileName+'.jpg'))
        img1 = img1[nCrop:,:,:]

        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2 = img1.copy()
        height,width,depth = img1.shape
        myMask = np.zeros((height,width), np.uint8)
        dst = img1.copy()
        output = np.zeros((height,width), np.uint8)

        hist1 = [1,2,3]
        hist2 = [2,3,4]

        fig1 = plt.figure()
        ax1 = plt.subplot2grid((3,1),(0,0))
        l1, = ax1.plot(hist1,color = 'blue')
        l2, = ax1.plot(hist2,color = 'red')
        ax2 = plt.subplot2grid((3,1),(1,0))
        l3, = ax2.plot(hist1,color = 'blue')
        l4, = ax2.plot(hist2,color = 'red')
        ax3 = plt.subplot2grid((3,1),(2,0))
        l5, = ax3.plot(hist1,color = 'blue')
        l6, = ax3.plot(hist2,color = 'red')


        #fig2 = plt.figure()
        #ax3d = fig2.add_subplot(111, projection='3d')


        histLines = [l1,l2,l3,l4,l5,l6]
        hist = [None] * 6

        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.imshow('image',img1)
        if not maskDraw:
            cv2.setMouseCallback('image',draw_circleSmall)
        else:
            cv2.setMouseCallback('image',draw_circle)
        cv2.resizeWindow('image', (width*4, height*4))

        while(True):
            #cv2.imshow('image',img1)
            alpha = 0.4
            cv2.addWeighted( img2, alpha, img1, 1.-alpha,0, dst);
            cv2.imshow('image',dst)
            masked_data = cv2.bitwise_and(img1, img1, mask=myMask)

            for i in range(3):
                hist[i*2] = cv2.calcHist([img1_hsv],[i],myMask,[256],[0,256])
                hist[i*2+1] = cv2.calcHist([img1_hsv],[i],1-myMask,[256],[0,256])



            cv2.imshow("masked", masked_data)
            if cv2.waitKey(20) & 0xFF == 27:
                break


        for i in range(len(histLines)):
            histLines[i].set_data(range(len(hist[i])),hist[i])

        ax1.set_ylim([0,max(max(hist[0]),max(hist[1]))])
        ax2.set_ylim([0,max(max(hist[2]),max(hist[3]))])
        ax3.set_ylim([0,max(max(hist[4]),max(hist[5]))])
        ax1.set_xlim([0,256])
        ax2.set_xlim([0,256])
        ax3.set_xlim([0,256])

    #cv2.destroyAllWindows()

        plt.show()
        cv2.imwrite((fileName+fileStub+'.jpg'),img2)
        np.save((fileName+fileStub+'.npy'),myMask)

        print('we got here after show')
        fileName = "testImg2.jpg"
        img1 = cv2.imread(fileName)
        cv2.inRange(img1_hsv, np.array([0, 0, 0]), np.array([255, 255, 110]), output)
        cv2.imshow("output", output)
        cv2.waitKey(0)

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    cd = False
    ld = False
    yld = False
    if args['--coneDraw']:
        cd = True
    if args['--lineDraw']:
        ld = True
    if args['--ylineDraw']:
        yld = True
    if args['--fileName']:
    #    tub = args['--tub']
    #    tubInd = args['--tubInd']
    #    nCrop = args['--nCrop']
        drawMask(args['--fileName'],coneDraw = cd,lineDraw = ld, ylineDraw = yld)
    if args['--fileName'] and args['--nCrop']:
        drawMask(args['--fileName'],int(args['--nCrop']),coneDraw = cd,lineDraw = ld, ylineDraw = yld)
