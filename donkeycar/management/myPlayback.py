#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it. 

Usage:
    test.py (playback) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--no_cache]
    test.py (edgeplayback) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--no_cache]
    test.py (bothplayback) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--no_cache]
    test.py (car) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--no_cache]
    test.py (singleShot) [--tub=<tub1,tub2,..tubn>] [--tubInd=<tubInd>]
    
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
from donkeycar.parts.camera import PiCamera
from donkeycar.parts.transform import Lambda

from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.parts.datastore import Tub, TubHandler, TubGroup
from donkeycar.parts.controller import LocalWebController, JoystickController
from donkeycar.parts.cv_pilot import clip
from donkeycar import utils
from jensoncal import lookAtFloorImg
from jensoncal import lookAtFloorImg2
from jensoncal import getEdges
print('imports done')

def playback(cfg, tub_names, model_name=None,doEdges = False,doBoth = False,doCar = False):
    if not tub_names:
        tub_names = os.path.join(cfg.DATA_PATH, '*')
    tubgroup = TubGroup(tub_names)
    
    if not model_name is None:
        from donkeycar.parts.keras import KerasCategorical
        kl = KerasCategorical()
        kl.load(model_name)
        pilot_angles = []
        pilot_throttles  = []
        
    
    
    print('tub_names', tub_names)
    
    tub_paths = utils.expand_path_arg(tub_names)
    print('TubGroup:tubpaths:', tub_paths)
    user_angles = []
    user_throttles  = []
    
        
    tubs = [Tub(path) for path in tub_paths]
    for tub in tubs:
            num_records = tub.get_num_records()
            print(num_records)
            for iRec in tub.get_index(shuffled=False):
                record = tub.get_record(iRec)
            #record = tubs.get_record(random.randint(1,num_records+1))
                img = record["cam/image_array"]
                user_angle = float(record["user/angle"])
                user_throttle = float(record["user/throttle"])
                user_angles.append(user_angle)
                user_throttles.append(user_throttle)
                if not model_name is None:
                    pilot_angle, pilot_throttle = kl.run(img)
                    pilot_angles.append(pilot_angle)
                    pilot_throttles.append(pilot_throttle)
            
    
    record = tubs[0].get_record(random.randint(1,num_records+1))
    user_angle = float(record["user/angle"])
    user_throttle = float(record["user/throttle"])
    print(img.shape)
    print('-----')
    print(user_angle)
    print(user_throttle)
    plt.figure()
    plt.imshow(img)
    plt.plot([80,80+10*user_throttle*np.cos(user_angle)],[120,120+100*user_throttle*np.sin(user_angle)])
    
    plt.figure()
    plt.plot(user_angles)
    plt.plot(user_throttles)
    
    
    
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((2,2),(0,0))
    record = tubs[0].get_record(1)
    img = record["cam/image_array"]
    imPlot = ax1.imshow(img,animated=True)
    edge1 = []
    edge1.append([])
    lchop = 60
    floorImg = lookAtFloorImg2(img,maxHeight = 0,balance = 1.0)[120:,:,:]
    #print(floorImg)
    ax3 = plt.subplot2grid((2,2),(0,1))
    if doEdges:
            imPlot2= ax3.imshow(getEdges(img[lchop: , :, :]))
    elif doBoth or doCar:
            temp = lookAtFloorImg2(img,balance = 1.0)[120:,:,:]
            edge1[0] = getEdges(temp)
            #imPlot2= ax3.imshow(temp)
            transformThenEdges = getEdges(temp)
            edgesThenTransform = lookAtFloorImg2(getEdges(img),balance = 1.0)[120:,:]
            transformThenEdges = cv2.normalize(transformThenEdges, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            edgesThenTransform = cv2.normalize(edgesThenTransform, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            gray = cv2.cvtColor(img[40:,:,:], cv2.COLOR_BGR2GRAY)
            gray = cv2.normalize(gray, np.array([]),alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            ret,gray = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)
            #gray = cv2.Canny(gray, 0, 200)
            imPlot3= ax3.imshow(gray,cmap='gray', alpha=1.0)
            edgeLine0, = ax3.plot([0,0],[0,1],color='green')
            edgeLine1, = ax3.plot([0,0],[0,1],color='green')
            edgeLine2, = ax3.plot([0,0],[0,1],color='green')
            edgeLine3, = ax3.plot([0,0],[0,1],color='green')
            edgeLine4, = ax3.plot([0,0],[0,1],color='green')
            edgeLine5, = ax3.plot([0,0],[0,1],color='green')
            edgeLine6, = ax3.plot([0,0],[0,1],color='green')
            edgeLine7, = ax3.plot([0,0],[0,1],color='green')
            edgeLine8, = ax3.plot([0,0],[0,1],color='green')
            edgeLine9, = ax3.plot([0,0],[0,1],color='green')
            edgeLine10, = ax3.plot([0,0],[0,1],color='green')
            edgeLine11, = ax3.plot([0,0],[0,1],color='green')
            
            steerLine, = ax1.plot([80,80],[0,80],color='orange')
            
            edgeLines = [edgeLine0,edgeLine1,edgeLine2,edgeLine3,edgeLine4,edgeLine5,edgeLine6,edgeLine7,edgeLine8,edgeLine9,edgeLine10,edgeLine11]
            #imPlot4= ax3.imshow(edgesThenTransform,cmap='gray', alpha=0.5)

            
    else:
            imPlot2= ax3.imshow(lookAtFloorImg2(img,balance = 1.0)[120:,:,:])
    
    ax2 = plt.subplot2grid((2,2),(1,0), colspan=2)
    line1, = ax2.plot(user_angles)
    line2, = ax2.plot(user_throttles)
    if not model_name is None:
        line4, = ax2.plot(pilot_angles)
        line5, = ax2.plot(pilot_throttles)
    line3, = ax2.plot([0,0],[-1,1])
    line4, = ax2.plot([0,0],[-1,1])
    
    allAngles = []
    allIntercepts = []
    
    def animate(i):
        record = tubs[0].get_record(i)
        img = record["cam/image_array"]
        imPlot.set_array(img)
        
        if doEdges:
            imPlot2.set_array(getEdges(img[40: , :, :]))
        elif doBoth:
            temp = lookAtFloorImg2(img,balance = 1.0)[120:,:,:]
            #imPlot2.set_array(temp)
            
            transformThenEdges = getEdges(temp)
            #edgesThenTransform = lookAtFloorImg2(getEdges(img),balance = 1.0)[120:,:]
            edgesThenTransform = (getEdges(img[lchop:,:,:]))
            #imPlot3.set_array((transformThenEdges+edgesThenTransform+edge1[0])/4)
            #gray = cv2.cvtColor(img[lchop:,:,:], cv2.COLOR_BGR2GRAY)
            gray = img[lchop:,:,2]
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            gray = cv2.normalize(gray,np.array([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            ret,gray = cv2.threshold(255-gray,255-70,255,cv2.THRESH_TRUNC)
            gray = 255-gray
            edges = cv2.Canny(gray, 0, 200)
            imPlot3.set_array(gray)
            #edge1[0] = (transformThenEdges+edgesThenTransform)
            #print("edges then transform")
            #print(edge1[0].shape)
            #print("transform then edges")
            #print(lookAtFloorImg2(getEdges(img),balance = 1.0)[120:,:].shape)
            
            minLineLength = 10
            maxLineGap = 20
            lines = cv2.HoughLinesP(edges,1,np.pi/180,30,np.array([]),minLineLength,maxLineGap)
            nPlot = 0
            for x in range(0, (len(edgeLines))):
                edgeLines[x].set_data([0,0],[0,0])
            myX = []
            myY = []
            myX1 = []
            myIntercept = []
            myGrad = []
            
            if lines is not None:
                print(len(lines))
                for x in range(0, (len(lines))):
  
                    for x1,y1,x2,y2 in lines[x]:
                        if (np.abs(y1-y2)>15) and nPlot<len(edgeLines):
                        
                            edgeLines[nPlot].set_data([x1,x2],[y1,y2])
                            myX.append(x2-x1)
                            myY.append(y1-y2)
                            myIntercept.append((x2*(y1-90)-x1*(y2-90))/(y1-y2))
                        
                            myGrad.append((x2-x1)/(y1-y2))
                        #print (min(y1,y2))
                            nPlot+=1
            
            #np.arctan2(y, x) * 180 / np.pi
                myAngle = np.arctan(np.median(myGrad))
                #print(myAngle)
                allAngles.append(myAngle)
                allIntercepts.append(np.median(myIntercept)/300)
                line3.set_data(range(len(allAngles)),allAngles)
                line4.set_data(range(len(allAngles)),allIntercepts)
            #ax2.set_ylim(min(allIntercepts),max(allIntercepts))
            #imPlot4.set_array(edgesThenTransform)
        elif doCar:
            edgesThenTransform =getEdges(img[50:,:,:])
            minLineLength = 10
            maxLineGap = 20
            imPlot3.set_array(edgesThenTransform)
            lines = cv2.HoughLinesP(edgesThenTransform,1,np.pi/180,25,np.array([]),minLineLength,maxLineGap)
            myX = []
            myY = []
            myX1 = []
            myIntercept = []
            myGrad = []
            nPlot = 0
            for x in range(0, (len(edgeLines))):
                edgeLines[x].set_data([0,0],[0,0])
            nLinesMax = 11
            myAngle=None
            interceptOut = None
            if lines is not None:
                print("----lines----")
                print(len(lines))
                for x in range(0, (len(lines))):
                    for x1,y1,x2,y2 in lines[x]:
                        if (np.abs(y1-y2)>10) and nPlot<nLinesMax:
                            edgeLines[nPlot].set_data([x1,x2],[y1,y2])
                            myX.append(x2-x1)
                            myY.append(y1-y2)
                            myIntercept.append((x2*(y1-90)-x1*(y2-90))/(y1-y2))                        
                            myGrad.append((x2-x1)/(y1-y2))
                            nPlot+=1
            
            #np.arctan2(y, x) * 180 / np.pi
                myAngle = np.arctan(np.median(myGrad))
                if np.isnan(myAngle): myAngle = None
                else: steerLine.set_data([80,80+40*np.sin(myAngle)],[80,80-40*np.cos(myAngle)])
                interceptOut = np.median(myIntercept)
                if np.isnan(interceptOut): interceptOut = None
            allAngles.append(myAngle)
            allIntercepts.append(np.median(myIntercept)/300)
            line3.set_data(range(len(allAngles)),allAngles)
            line4.set_data(range(len(allAngles)),allIntercepts)
       
        else:
            imPlot2.set_array(lookAtFloorImg2(img,balance = 1.0)[120:,:,:])
        
        #print(i)
        #sys.stdout.flush(
        return imPlot,


    # Init only required for blitting to give a clean slate.
    def init():
        record = tubs[0].get_record(1)
        img #= record["cam/image_array"]
        imPlot.set_array(img)
        line3.set_data([0,0],[-1,1])
        return imPlot,
    
    
    
    ani = animation.FuncAnimation(fig, animate, np.arange(1, tubs[0].get_num_records()),
                              interval=100, blit=False)
                              
    plt.show()
    
    #user_angles.append(user_angle)
    #user_throttles.append(user_throttle)
    #pilot_angles.append(pilot_angle)
    #pilot_throttles.append(pilot_throttle)
            

def playback1shot(cfg, tub, tubInd, doEdges=True):
    paths = glob.glob(tub+'/*.jpg')
    print(tub)
    print(int(tubInd))
    #print(paths)
    sys.stdout.flush()
    fig = plt.figure()
    raw = cv2.imread(paths[int(tubInd)])
    print(raw.dtype)
    edgesThenTransform = lookAtFloorImg2(getEdges(raw),balance = 1.0)[120:,:]
    print(edgesThenTransform.dtype)
    print('max')
    print(edgesThenTransform.max())
    print('min')
    print(edgesThenTransform.min())
    print(raw.shape)
    ax1 = plt.subplot2grid((5,1),(0,0))
    imPlot = ax1.imshow(raw)
    ax2 = plt.subplot2grid((5,1),(1,0))
    imPlot2 = ax2.imshow(edgesThenTransform)
    ax3 = plt.subplot2grid((5,1),(2,0))
    imPlot3 = ax3.imshow(edgesThenTransform,cmap='gray')
    minLineLength = 30
    maxLineGap = 10
    lines = cv2.HoughLinesP(edgesThenTransform,1,np.pi/180,40,minLineLength,maxLineGap)
    for x in range(0, (len(lines))):
        for x1,y1,x2,y2 in lines[x]:
            if (np.abs(y1-y2)>10):
                ax3.plot([x1,x2],[y1,y2])
    ax4 = plt.subplot2grid((5,1),(3,0))
    
    for x in range(0, (len(lines))):
        for x1,y1,x2,y2 in lines[x]:
            ax4.plot(x,np.abs(y1-y2),marker='o') 
            
    
    tub_paths = utils.expand_path_arg(tub)
    tubs = [Tub(path) for path in tub_paths]
    record = tubs[0].get_record(tubInd)   
    raw2 = record["cam/image_array"] 
    ax5 = plt.subplot2grid((5,1),(4,0))
    edgesThenTransform2 = lookAtFloorImg2(getEdges(raw2),balance = 1.0)[120:,:]
    imPlot5 = ax5.imshow(edgesThenTransform2)
    lines2 = cv2.HoughLinesP(edgesThenTransform2,1,np.pi/180,40,minLineLength,maxLineGap)
    for x in range(0, (len(lines2))):
        for x1,y1,x2,y2 in lines2[x]:
            if (np.abs(y1-y2)>10):
                ax5.plot([x1,x2],[y1,y2])
    print(raw.dtype)  
    
    plt.show()

if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    print(args)
    if args['playback']:
        tub = args['--tub']
        model = args['--model']
        cache = not args['--no_cache']
        playback(cfg, tub, model)
    if args['edgeplayback']:
        tub = args['--tub']
        model = args['--model']
        cache = not args['--no_cache']
        playback(cfg, tub, model, doEdges=True)
    if args['bothplayback']:
        tub = args['--tub']
        model = args['--model']
        cache = not args['--no_cache']
        playback(cfg, tub, model, doEdges=False,doBoth=True)
    if args['car']:
        tub = args['--tub']
        model = args['--model']
        cache = not args['--no_cache']
        playback(cfg, tub, model, doEdges=False,doBoth=False,doCar=True)
    if args['singleShot']:
        tub = args['--tub']
        tubInd = args['--tubInd']
        cache = not args['--no_cache']
        playback1shot(cfg, tub, tubInd, doEdges=True)


