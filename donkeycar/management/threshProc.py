#!/usr/bin/env python3
"""
Look at a tub and process it based on thresholds and extracting road markers

Usage:
    test.py (thresh1) [--tub=<tub1,tub2,..tubn>] [--tubInd=<tubInd>] [--nCrop=<n1>]
    test.py (thresh1HSV) [--tub=<tub1,tub2,..tubn>] [--tubInd=<tubInd>] [--nCrop=<n1>]
    test.py (threshHSV) [--tub=<tub1,tub2,..tubn>] [--nCrop=<n1>]
    test.py (thresh) [--tub=<tub1,tub2,..tubn>] [--nCrop=<n1>] [--nThresh=<n1>]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
    --js             Use physical joystick.
"""

import os
import sys
from docopt import docopt
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import donkeycar as dk
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, TextBox
import numpy as np
import numpy.polynomial.polynomial as poly
import random
import glob
#import parts
from donkeycar.parts.datastore import Tub, TubHandler, TubGroup
from donkeycar import utils
from donkeycar.management.jensoncal import lookAtFloorImg
from donkeycar.management.jensoncal import lookAtFloorImg2
from donkeycar.management.jensoncal import getEdges
from donkeycar.management.image_thresholding.get_Markers import findLineMarkers
from donkeycar.management.image_thresholding.analyse_masks import runClfImg
from sklearn.svm import SVC
from sklearn.externals import joblib
#import tensorflow as tf
print('imports done')


#sess = tf.Session()
#new_saver = tf.train.import_meta_graph('my_test_model.meta')
#new_saver.restore(sess, tf.train.latest_checkpoint('./'))
#M = np.array([[3.48407563   4.68019348  -193.634274], [ -0.0138199636 4.47986683 -38.3089390], [0.0 0.0 1.0]])
#M=np.array([[-1.3026235852665167, -3.499123877776032, 245.75446559156023], [-0.03176219298555294, -5.213807674195841, 254.91345254435038], [-0.000211747953236998, -0.02383023318793577, 1.0]])
#M=np.array([[3.4840756328915137, 4.680193479489915, -193.6342735096424], [-0.013819963565551818, 4.479866825805638, -38.308939003706215], [-0.0009213309043700334, 0.06172917059279264, 1.0]])
#M=np.array([[2.6953954394120174, 4.212827438909477, -140.8803316791254], [-0.01381996356555254, 4.479866825805639, -38.308939003706115], [-0.0009213309043700707, 0.061729170592792676, 1.0]])
M=np.array([[0.7288447030443173, 5.383427898971357, 9.457197002209224], [-5.551115123125783e-16, 2.8193645731219714, 3.8191672047105385e-14], [-0.0017493890126173033, 0.06997556050469123, 1.0]])

def getThresh(raw,threshLevel=127):
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    #gray = cv2.normalize(gray, np.array([]),alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    ret,threshed = cv2.threshold(gray,threshLevel,1,cv2.THRESH_BINARY_INV)
    #threshed = 255-threshed
    return threshed
def getThreshRGB(raw,lowerRGB=np.array([50, 0, 0]), upperRGB=np.array([255, 150, 150]),):
    threshed = cv2.inRange(raw, lowerRGB, upperRGB)
    return threshed
def getThreshHSV(raw,lowerHSV, upperHSV, oneshot = False):
    if oneshot:
        hsvImg = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB)
    else:
        hsvImg = cv2.cvtColor(raw, cv2.COLOR_RGB2LAB)
    #gray = cv2.normalize(gray, np.array([]),alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    hsvImg = cv2.bilateralFilter(hsvImg, 11, 17, 17)
    #ret,threshed = cv2.threshold(gray,hsvImg,1,cv2.THRESH_BINARY_INV)
    #threshed  = # apply threshold on hsv image
    threshed = cv2.inRange(hsvImg, lowerHSV, upperHSV)
    #threshed = 255-threshed
    return threshed,hsvImg
def getPolyFit(threshImg):
    #means = np.true_divide(threshImg.sum(1),(threshImg!=0).sum(1)+0.0001)/2.
    sums = threshImg.sum(1)
    pointsX = []
    pointsY = []
    #pointsXT = []
    #pointsYT = []
    y = 0
    for s in sums:
        if s>2:#lower[2]/2:
            pointsY.append(y)
            x = np.median(np.nonzero(threshImg[y,:]))
            pointsX.append(x)
            #original = np.array([[x, y]], dtype=np.float32)
            #original = np.array([original])
            #converted = cv2.perspectiveTransform(original, M)
            #pointsXT.append(converted[0][0][0])
            #pointsYT.append(converted[0][0][1])
        y+=1
    coeffs = []
    if len(pointsX)>3:
        coeffs = poly.polyfit(pointsY, pointsX, 2)
        print(coeffs)
        #b = poly.poly1d(a)
        #fitLine.set_data(b(pointsY),pointsY)
        #aT = np.polyfit(pointsYT, pointsXT, 2)
        #bT = np.poly1d(aT)
        #fitLineT.set_data(bT(pointsYT),pointsYT)
    return coeffs

def imProc(raw,nCrop = 45,threshLevel=127):
    raw = raw[nCrop:,:,:]
    edgesImg=getEdges(raw)
    threshed = getThresh(raw,threshLevel)
    return edgesImg,threshed

def doThresh(cfg, tub_names,nCrop = 45,nThresh = 127,limit_axes = True):
    clf_cone = joblib.load('image_thresholding/cone_RGB_000.pkl')
    clf_track = joblib.load('image_thresholding/track_HSV_000.pkl')
    nCrop = int(nCrop)
    tubgroup = TubGroup(tub_names)
    tub_paths = utils.expand_path_arg(tub_names)
    tubs = [Tub(path) for path in tub_paths]
    tub = tubs[0]
    tubInds = tub.get_index(shuffled=False)
    kTub = 0
    nRecords = len(tubInds)
    record = tub.get_record(tubInds[kTub])
    raw = record["cam/image_array"]
    print(raw.shape)
    edgesImg,threshed = imProc(raw,nCrop)
    filledXY, filledA,allXY, allA = findLineMarkers(raw,nCrop,nThresh,doPlot = False)
    trackImg = runClfImg(raw[nCrop:,:,:],clf_track)
    coneImg = runClfImg(raw[nCrop:,:,:],clf_cone)
    fig = plt.figure()
    if limit_axes:
        ax1 = plt.subplot2grid((1,2),(0,0))

        ax5 = plt.subplot2grid((1,2),(0,1))
    else:
        ax1 = plt.subplot2grid((3,2),(0,0))
        ax2 = plt.subplot2grid((3,2),(1,0))
        #imPlot2 = ax2.imshow(edgesImg)
        ax3 = plt.subplot2grid((3,2),(2,0))
        ax4 = plt.subplot2grid((3,2),(2,1))
        ax5 = plt.subplot2grid((3,2),(0,1))
        ax6 = plt.subplot2grid((3,2),(1,1))
    imPlot1 = ax1.imshow(raw[nCrop:,:,:])
    # make a color map of fixed colors
    cmap = mpl.colors.ListedColormap(['black', 'xkcd:dark gray','xkcd:gray','xkcd:orange'])
    bounds=[0,0.5,3,6,20]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # tell imshow about color map so that only set colors are used
    trackPlot = ax5.imshow(trackImg*6+coneImg*10, interpolation='nearest',
                    cmap=cmap, norm=norm)

    #trackPlot = ax5.imshow(trackImg*0.5,cmap="gray",vmax = 1,vmin = 0)
    #conePlot = ax6.imshow(coneImg)
    if limit_axes==False:
        imPlot3 = ax2.imshow(threshed)
        imPlot2 = ax2.imshow(edgesImg,alpha=0.5)
    myAlpha=0.2
    cFilled = 'xkcd:cream'
    cAll = 'xkcd:blue grey'
    allPlot, = ax5.plot(allXY[0],allXY[1],linestyle = 'None',marker = '.',color = cAll)
    filledPlot, = ax5.plot(filledXY[0],filledXY[1],linestyle = 'None',marker = '.',color = cFilled)
    if limit_axes==False:
        allPlotX, = ax3.plot(allXY[0],allXY[1],linestyle = 'None',marker = '.',alpha = myAlpha,color = cAll)
        filledPlotX, = ax3.plot(filledXY[0],filledXY[1],linestyle = 'None',marker = '.',alpha = myAlpha,color = cFilled)
        filledPlotY, = ax4.plot(filledXY[1],filledA,linestyle = 'None',marker = '.',alpha = myAlpha,color = cFilled)
        allPlotY, = ax4.plot(allXY[1],allA,linestyle = 'None',marker = '.',alpha = myAlpha,color = cAll)

    height,width = threshed.shape
    notPatchA = []
    notPatchXY = [[],[]]
    isPatchA = []
    isPatchXY = [[],[]]
    isPatch2A = []
    isPatch2XY = [[],[]]


    if True:
        def animate(i):
            record = tub.get_record(tubInds[i])
            raw = record["cam/image_array"]
            imPlot1.set_data(raw[nCrop:,:,:])
            trackImg = runClfImg(raw[nCrop:,:,:],clf_track)
            coneImg = runClfImg(raw[nCrop:,:,:],clf_cone)
            edgesImg,threshed = imProc(raw,nCrop,int(nThresh))

            filledXY, filledA,allXY, allA = findLineMarkers(raw,nCrop,nThresh,doPlot = False)
            truFilledA = []
            truFilledXY = [[],[]]
            for fa,fx,fy in zip(filledA,filledXY[0],filledXY[1]):

                if fa>3 and fy>3:
                    truFilledA.append(fa)
                    truFilledXY[0].append(fx)
                    truFilledXY[1].append(fy)
            if len(allA)>0:
                notPatchA.extend(allA)
                notPatchXY[0].extend(allXY[0])
                notPatchXY[1].extend(allXY[1])
            if len(filledA)>0:
                isPatchA.extend(filledA)
                isPatchXY[0].extend(filledXY[0])
                isPatchXY[1].extend(filledXY[1])
            if len(truFilledA)>0:
                isPatch2A.extend(truFilledA)
                isPatch2XY[0].extend(truFilledXY[0])
                isPatch2XY[1].extend(truFilledXY[1])


            if limit_axes==False:
                imPlot2.set_data(edgesImg)
                imPlot3.set_data(threshed)
                allPlotX.set_data(notPatchXY[1],notPatchA)
                filledPlotX.set_data(isPatchXY[1],isPatchA)
                allPlotY.set_data(notPatchXY[1],notPatchA)
                filledPlotY.set_data(isPatchXY[1],isPatchA)
                ax3.set_ylim(0,50)#max(max(notPatchA),max(notPatchA))*1.2)
                ax4.set_ylim(0,50)#max(max(notPatchA),max(notPatchA))*1.2)
                ax3.set_xlim(0,height)
                ax4.set_xlim(0,height)
            allPlot.set_data(filledXY[0],filledXY[1])
            filledPlot.set_data(truFilledXY[0],truFilledXY[1])
            trackPlot.set_data(threshed*1+trackImg*2+coneImg*10)

            return imPlot1,
        ani = animation.FuncAnimation(fig, animate, np.arange(1, nRecords),
                                  interval=100, blit=False)
    plt.show()


def doThresh1shot(cfg, tub, tubInd,nCrop = 45,nThresh = 127):
    paths = glob.glob(tub+'/*.jpg')
    print(tub)
    print(int(tubInd))
    nCrop = int(nCrop)
    #print(paths)
    sys.stdout.flush()
    fig = plt.figure()
    raw = cv2.imread(paths[int(tubInd)])
    edgesImg,threshed = imProc(raw,nCrop)
    ax1 = plt.subplot2grid((2,2),(0,0))
    imPlot = ax1.imshow(raw)
    ax2 = plt.subplot2grid((2,2),(1,0))
    imPlot = ax2.imshow(edgesImg)
    ax3 = plt.subplot2grid((2,2),(0,1))
    imPlot = ax3.imshow(threshed)
    ax2 = plt.subplot2grid((2,2),(1,0))
    imPlot = ax2.imshow(edgesImg)

    plt.show()


def doThresh1shotHSV(cfg, tub, tubInd,nCrop = 45,nThresh = 127):
    paths = glob.glob(tub+'/*.jpg')
    print(tub)
    print(int(tubInd))
    nCrop = int(nCrop)
    #print(paths)
    sys.stdout.flush()
    fig = plt.figure()
    raw = cv2.imread(paths[int(tubInd)])
    raw = raw[nCrop:,:,:]
    #edgesImg,threshed = imProc(raw,nCrop)
    lower = np.array([170, 0, 150])
    upper = np.array([255, 120, 255])
    threshed,hsvImg = getThreshHSV(raw,lower,upper,oneshot = True)
    ax1 = plt.subplot2grid((3,2),(0,0))
    imPlot = ax1.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    ax2 = plt.subplot2grid((3,2),(0,1))
    imPlot = ax2.imshow(np.squeeze(hsvImg[:,:,0]))
    ax3 = plt.subplot2grid((3,2),(1,0))
    imPlot = ax3.imshow(np.squeeze(hsvImg[:,:,1]))
    ax4 = plt.subplot2grid((3,2),(1,1))
    imPlot = ax4.imshow(np.squeeze(hsvImg[:,:,2]))
    ax5 = plt.subplot2grid((3,2),(2,1))
    imPlot = ax5.imshow(threshed)
    means = np.true_divide(threshed.sum(1),(threshed!=0).sum(1)+0.0001)/2.
    meanlines = ax5.plot(means,range(len(means)))

    #ax3 = plt.subplot2grid((2,2),(0,1))
    #imPlot = ax3.imshow(threshed)
    #ax2 = plt.subplot2grid((2,2),(1,0))
    #imPlot = ax2.imshow(edgesImg)

    plt.show()
def doThreshHSV(cfg, tub_names,nCrop = 45):
    from threshClass import Thresh
    tubgroup = TubGroup(tub_names)
    tub_paths = utils.expand_path_arg(tub_names)
    tubs = [Tub(path) for path in tub_paths]
    tub = tubs[0]
    tubInds = tub.get_index(shuffled=False)
    kTub = 0
    nRecords = len(tubInds)
    record = tub.get_record(tubInds[kTub])
    raw = record["cam/image_array"]
    raw = raw[nCrop:,:,:]
    #edgesImg,threshed = imProc(raw,nCrop)
    #lower = np.array([20, 60, 120])
    #upper = np.array([40, 255, 255])
    lowerRGB = np.array([20, 60, 120])
    upperRGB = np.array([40, 255, 255])
    #threshHSV = Thresh(np.array([20, 60, 120]),np.array([40, 255, 255]))
    lower = np.array([120, 0, 147])
    upper = np.array([245, 130, 255])
    threshHSV = Thresh(lower,upper)
    threshRGB = Thresh(np.array([120, 120, 120]),np.array([255, 255, 255]))
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,3),(0,0))
    imPlot1 = ax1.imshow(raw)
    hsvImg,orig = getThreshHSV(raw,lower,upper)
    ax2 = plt.subplot2grid((1,3),(0,1))
    imPlot2 = ax2.imshow(hsvImg)
    means = np.true_divide(hsvImg.sum(1),(hsvImg!=0).sum(1)+0.0001)/2.
    meanlines, = ax2.plot(means,range(len(means)))
    threshLine, = ax2.plot([75,75],[0,75],color = 'r',linestyle = '-.')
    pointsLine, = ax2.plot([75,75],[0,75],color = 'xkcd:orange',linestyle = '',marker = '.')
    fitLine, = ax2.plot([75,75],[0,75],color = 'xkcd:yellow',linestyle = '-.')
    frametext = ax1.text(0.1, 0.1, '1', horizontalalignment='center',
        verticalalignment='center', transform=ax1.transAxes,color='xkcd:light blue')
    axSliderFrame = plt.axes([0.2, 0.04, 0.4, 0.03], facecolor='gray')

    ax3 = plt.subplot2grid((1,3),(0,2))
    rows,cols = hsvImg.shape
    hsvImgT = cv2.warpPerspective(hsvImg,M,(cols,rows))
    imPlot3 = ax3.imshow(hsvImgT)
    pointsLineT, = ax3.plot([0,0],[0,0],color = 'xkcd:orange',linestyle = '',marker = '.')
    fitLineT, = ax3.plot([75,75],[0,75],color = 'xkcd:yellow',linestyle = '-.')
    ax3.set_xlim([40,120])
    ax3.set_ylim([40,10])
    sFrame = Slider(axSliderFrame, 'Frame no.', 0,nRecords, valinit=0,valfmt='%d')
    tbHSV = []
    tbRGB = []
    for i in range(6):
        axbox = plt.axes([0.1+i*0.12, 0.25, 0.1, 0.075])
        axboxRGB = plt.axes([0.1+i*0.12, 0.15, 0.1, 0.075])
        if i==0:
            titleStr = 'HSV'
            titleStrRGB = 'RGB'
        else:
            titleStr = ''
            titleStrRGB = ''
        tbHSV.append(TextBox(axbox, titleStr, initial=str(np.concatenate((threshHSV.lower, threshHSV.upper), axis=0)[i])))
        tbRGB.append(TextBox(axboxRGB, titleStrRGB, initial=str(np.concatenate((threshRGB.lower, threshRGB.upper), axis=0)[i])))


    frameNo = [0]
    frameCurrent = [0]
    def updateFrameSlider(val):
        frameNo[0] = int(sFrame.val)
        frameCurrent[0] = 0
    sFrame.on_changed(updateFrameSlider)

    def animate(i):
        actualFrame = (frameCurrent[0]+frameNo[0])%nRecords
        sFrame.vline.set_xdata([actualFrame,actualFrame])
        frametext.set_text(str(tubInds[actualFrame]))
        frameCurrent[0]+=1
        record = tub.get_record(tubInds[actualFrame])
        raw = record["cam/image_array"]
        raw = raw[nCrop:,:,:]
        #lower,upper = getThresh(tbHSV)
        #lowerRGB,upperRGB = getThresh(tbRGB)
        hsvImg,orig = getThreshHSV(raw,lower,upper)
        imPlot1.set_data(raw)
        bChannel = np.squeeze(orig[:,:,2])
        hsvImg = cv2.bitwise_and(bChannel,bChannel,mask = hsvImg)
        imPlot2.set_data(hsvImg)
        means = np.true_divide(hsvImg.sum(1),(hsvImg!=0).sum(1)+0.0001)/2.
        pointsX = []
        pointsY = []
        pointsXT = []
        pointsYT = []
        y = 0
        for m in means:
            if m>75:#lower[2]/2:
                pointsY.append(y)
                x = np.argmax(hsvImg[y,:])
                pointsX.append(x)
                original = np.array([[x, y]], dtype=np.float32)
                original = np.array([original])
                converted = cv2.perspectiveTransform(original, M)
                pointsXT.append(converted[0][0][0])
                pointsYT.append(converted[0][0][1])
            y+=1


        pointsLine.set_data(pointsX,pointsY)
        pointsLineT.set_data(pointsXT,pointsYT)
        if len(pointsX)>3:
            a = np.polyfit(pointsY, pointsX, 2)
            b = np.poly1d(a)
            fitLine.set_data(b(pointsY),pointsY)
            aT = np.polyfit(pointsYT, pointsXT, 2)
            bT = np.poly1d(aT)
            fitLineT.set_data(bT(pointsYT),pointsYT)

        meanlines.set_data(means,range(len(means)))
        hsvImgT = cv2.warpPerspective(hsvImg,M,(cols,rows))
        imPlot3.set_data(hsvImgT)
    ani = animation.FuncAnimation(fig, animate, np.arange(1, nRecords),
                              interval=100, blit=False)
    plt.show()


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    print(args)
    if args['thresh1']:
        tub = args['--tub']
        tubInd = args['--tubInd']
        nCrop = args['--nCrop']
        doThresh1shot(cfg, tub, tubInd,nCrop)
    if args['thresh1HSV']:
        tub = args['--tub']
        tubInd = args['--tubInd']
        nCrop = args['--nCrop']
        #nThresh = args['--nThresh']
        doThresh1shotHSV(cfg, tub, tubInd,nCrop)
    if args['threshHSV']:
        tub = args['--tub']
        #nCrop = args['--nCrop']
        #nThresh = args['--nThresh']
        doThreshHSV(cfg, tub)
    if args['thresh']:
        tub = args['--tub']
        nCrop = args['--nCrop']
        #nThresh = args['--nThresh']
        doThresh(cfg, tub, nCrop, int(nThresh))
