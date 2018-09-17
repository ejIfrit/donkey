#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    test.py (thresh1) [--tub=<tub1,tub2,..tubn>] [--tubInd=<tubInd>] [--nCrop=<n1>]
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
import numpy as np
import random
import glob
#import parts
from donkeycar.parts.datastore import Tub, TubHandler, TubGroup
from donkeycar import utils
from jensoncal import lookAtFloorImg
from jensoncal import lookAtFloorImg2
from jensoncal import getEdges
print('imports done')

#M = np.array([[3.48407563   4.68019348  -193.634274], [ -0.0138199636 4.47986683 -38.3089390], [0.0 0.0 1.0]])
#M=np.array([[-1.3026235852665167, -3.499123877776032, 245.75446559156023], [-0.03176219298555294, -5.213807674195841, 254.91345254435038], [-0.000211747953236998, -0.02383023318793577, 1.0]])
#M=np.array([[3.4840756328915137, 4.680193479489915, -193.6342735096424], [-0.013819963565551818, 4.479866825805638, -38.308939003706215], [-0.0009213309043700334, 0.06172917059279264, 1.0]])
#M=np.array([[2.6953954394120174, 4.212827438909477, -140.8803316791254], [-0.01381996356555254, 4.479866825805639, -38.308939003706115], [-0.0009213309043700707, 0.061729170592792676, 1.0]])
M=np.array([[0.7288447030443173, 5.383427898971357, 9.457197002209224], [-5.551115123125783e-16, 2.8193645731219714, 3.8191672047105385e-14], [-0.0017493890126173033, 0.06997556050469123, 1.0]])

def getThresh(raw,threshLevel=127):
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    #gray = cv2.normalize(gray, np.array([]),alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    ret,threshed = cv2.threshold(gray,threshLevel,255,cv2.THRESH_TOZERO)
    return threshed

def imProc(raw,nCrop = 45,threshLevel=127):
    raw = raw[nCrop:,:,:]
    edgesImg=getEdges(raw)
    threshed = getThresh(raw,threshLevel)
    return edgesImg,threshed

def doThresh(cfg, tub_names,nCrop = 45,nThresh = 127):
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
    fig = plt.figure()
    ax1 = plt.subplot2grid((3,1),(0,0))
    imPlot1 = ax1.imshow(raw[nCrop:,:,:])
    ax2 = plt.subplot2grid((3,1),(1,0))
    #imPlot2 = ax2.imshow(edgesImg)
    ax3 = plt.subplot2grid((3,1),(2,0))
    imPlot3 = ax2.imshow(threshed)
    imPlot2 = ax2.imshow(edgesImg,alpha=0.5)
    def animate(i):
        record = tub.get_record(tubInds[i])
        raw = record["cam/image_array"]
        edgesImg,threshed = imProc(raw,nCrop,int(nThresh))

        imPlot1.set_data(raw[nCrop:,:,:])
        imPlot2.set_data(edgesImg)
        imPlot3.set_data(threshed)
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

if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    print(args)
    if args['thresh1']:
        tub = args['--tub']
        tubInd = args['--tubInd']
        nCrop = args['--nCrop']
        doThresh1shot(cfg, tub, tubInd,nCrop)
    if args['thresh']:
        tub = args['--tub']
        nCrop = args['--nCrop']
        nThresh = args['--nThresh']
        doThresh(cfg, tub, nCrop, nThresh)
