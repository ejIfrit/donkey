#!/usr/bin/env python3
"""
Look at a tub and process it based on thresholds and extracting road markers

Usage:
    test.py (thresh1) [--tub=<tub1,tub2,..tubn>] [--tubInd=<tubInd>] [--nCrop=<n1>]
    test.py (segm) [--tub=<tub1,tub2,..tubn>] [--nCrop=<n1>] [--nThresh=<n1>] [--modelDir=<d1>]

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
from matplotlib.widgets import Slider
import numpy as np
import random
import glob
#import parts
from donkeycar.parts.datastore import Tub, TubHandler, TubGroup
from donkeycar import utils
from jensoncal import lookAtFloorImg
from jensoncal import lookAtFloorImg2
from jensoncal import getEdges
from image_thresholding.get_Markers import findLineMarkers
from image_thresholding.analyse_masks import runClfImg
from sklearn.svm import SVC
from sklearn.externals import joblib
import tensorflow as tf
from threshProc import imProc
from segm.model import Model_fromLoad
print('imports done')

clf_cone = joblib.load('image_thresholding/cone_RGB_000.pkl')
clf_track = joblib.load('image_thresholding/track_HSV_000.pkl')
batch_size = 1
dropout = 0.7
sess = tf.Session()
sess.run(tf.global_variables_initializer())
graph = tf.get_default_graph()

#M = np.array([[3.48407563   4.68019348  -193.634274], [ -0.0138199636 4.47986683 -38.3089390], [0.0 0.0 1.0]])
#M=np.array([[-1.3026235852665167, -3.499123877776032, 245.75446559156023], [-0.03176219298555294, -5.213807674195841, 254.91345254435038], [-0.000211747953236998, -0.02383023318793577, 1.0]])
#M=np.array([[3.4840756328915137, 4.680193479489915, -193.6342735096424], [-0.013819963565551818, 4.479866825805638, -38.308939003706215], [-0.0009213309043700334, 0.06172917059279264, 1.0]])
#M=np.array([[2.6953954394120174, 4.212827438909477, -140.8803316791254], [-0.01381996356555254, 4.479866825805639, -38.308939003706115], [-0.0009213309043700707, 0.061729170592792676, 1.0]])
M=np.array([[0.7288447030443173, 5.383427898971357, 9.457197002209224], [-5.551115123125783e-16, 2.8193645731219714, 3.8191672047105385e-14], [-0.0017493890126173033, 0.06997556050469123, 1.0]])

def prepForSegm(img1_orig,nCrop=45,nResize=0.5,goGrey=True,addYCoords = True):
    if nCrop>0:
        o = img1_orig[nCrop:,:,:]
    if nResize!=1:
        height,width,depth = o.shape
        o = cv2.resize(o,(int(width*nResize),int(height*nResize)))

    if goGrey==True:
        o = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY)
    else:
        o = cv2.cvtColor(o, cv2.COLOR_BGR2RGB)
    w = o.shape[0]
    h=o.shape[1]
    x = np.linspace(0., 255., h)
    y = np.linspace(0., 255., w)
    temp, yv = np.meshgrid(x, y)
    o = np.stack((o,yv),axis=-1)
    o=(o-130)*(1./70.)
    return o


def doSegm(cfg,tub_names,nCrop=45,nThresh = 127,modelDir=''):
    modelPath = 'models/'+modelDir+'/'
    modelToLoad = glob.glob(modelPath+'/'+'*.meta')
    new_saver = tf.train.import_meta_graph(modelToLoad[0])
    new_saver.restore(sess, tf.train.latest_checkpoint(modelPath))
    model = Model_fromLoad(graph,1,dropout,w = 37,h=80,ndims = 2)
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
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,3),(0,0))
    ax3 = plt.subplot2grid((1,3),(0,1))
    ax2 = plt.subplot2grid((1,3),(0,2))
    o = prepForSegm(raw)
    o = o[np.newaxis,:,:,:]
    imPlot1 = ax1.imshow(np.squeeze(o[0,:,:,0]))
    print(sess.run(graph.get_tensor_by_name("bias1:0")))
    print(sess.run(graph.get_tensor_by_name("weight1:0"))[0,0,0,:])
    segm_map_pred_load = sess.run(model.h4, feed_dict={model.image:o})
    filledXY, filledA,allXY, allA = findLineMarkers(raw,nCrop,nThresh,doPlot = False)
    trackImg = runClfImg(raw[nCrop:,:,:],clf_track)
    coneImg = runClfImg(raw[nCrop:,:,:],clf_cone)
    edgesImg,threshed = imProc(raw,nCrop)
    cmap = mpl.colors.ListedColormap(['black', 'xkcd:dark gray','xkcd:gray','xkcd:orange'])
    bounds=[0,0.5,3,6,20]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # tell imshow about color map so that only set colors are used
    trackPlot = ax2.imshow(threshed*1+trackImg*2+coneImg*10, interpolation='nearest',
                    cmap=cmap, norm=norm)
    cFilled = 'xkcd:cream'
    cAll = 'xkcd:blue grey'
    allPlot, = ax2.plot(allXY[0],allXY[1],linestyle = 'None',marker = '.',color = cAll)
    filledPlot, = ax2.plot(filledXY[0],filledXY[1],linestyle = 'None',marker = '.',color = cFilled)
    frametext = ax1.text(0.1, 0.1, '1', horizontalalignment='center',
        verticalalignment='center', transform=ax1.transAxes,color='xkcd:light blue')
    height,width = threshed.shape
    notPatchA = []
    notPatchXY = [[],[]]
    isPatchA = []
    isPatchXY = [[],[]]
    isPatch2A = []
    isPatch2XY = [[],[]]
    segmPlot = ax3.imshow(np.squeeze(segm_map_pred_load),cmap='gray',vmin=0., vmax=255./145.)

    axSliderFrame = plt.axes([0.2, 0.04, 0.4, 0.03], facecolor='gray')
    sFrame = Slider(axSliderFrame, 'Frame no.', 0,nRecords, valinit=0,valfmt='%d')
    frameNo = [0]
    frameCurrent = [0]
    def updateFrameSlider(val):
        frameNo[0] = int(sFrame.val)
        frameCurrent[0] = 0
    sFrame.on_changed(updateFrameSlider)


    if True:
        def animate(i):
            actualFrame = (frameCurrent[0]+frameNo[0])%nRecords
            sFrame.vline.set_xdata([actualFrame,actualFrame])
            frameCurrent[0]+=1
            record = tub.get_record(tubInds[actualFrame])
            frametext.set_text(str(tubInds[actualFrame]))
            raw = record["cam/image_array"]
            o = prepForSegm(raw)
            o = o[np.newaxis,:,:,:]
            segm_map_pred = sess.run(model.h4, feed_dict={model.image:o})
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

            allPlot.set_data(filledXY[0],filledXY[1])
            filledPlot.set_data(truFilledXY[0],truFilledXY[1])
            trackPlot.set_data(threshed*1+trackImg*2+coneImg*10)
            segmPlot.set_data(np.squeeze(segm_map_pred))
            return imPlot1,
        ani = animation.FuncAnimation(fig, animate, np.arange(1, nRecords),
                                  interval=100, blit=False)
    plt.show()

def doThresh(cfg, tub_names,nCrop = 45,nThresh = 127,limit_axes = True):
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
    trackPlot = ax5.imshow(threshed*1+trackImg*2+coneImg*10, interpolation='nearest',
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
    pass

if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    print(args)
    if args['thresh1']:
        tub = args['--tub']
        tubInd = args['--tubInd']
        nCrop = args['--nCrop']
        doThresh1shot(cfg, tub, tubInd,nCrop)
    if args['segm']:
        tub = args['--tub']
        nCrop = args['--nCrop']
        nThresh = args['--nThresh']
        modelDir = args['--modelDir']
        doSegm(cfg, tub, nCrop, int(nThresh),modelDir)
