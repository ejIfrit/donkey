#!/usr/bin/env python3
"""
playback a tub and show what the image procesisng is doing,
and what the outputs look linke
"""

import os
import sys
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import donkeycar as dk
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import numpy as np
#import parts
from donkeycar.parts.datastore import Tub, TubHandler, TubGroup
from donkeycar.parts.cv_pilot import pilotLF
from donkeycar.management.jensoncal import getEdges
from donkeycar import utils
print('imports done')

# an array that transforms the camera image to make vertical parallel lines appear so
M=np.array([[0.7288447030443173, 5.383427898971357, 9.457197002209224], [-5.551115123125783e-16, 2.8193645731219714, 3.8191672047105385e-14], [-0.0017493890126173033, 0.06997556050469123, 1.0]])
# TODO: add in thresholding, semantic segmentation and NN visualisation
class sliderPlayBack(object):
    """a self-contained slider class for playback
        using the matplotlib slider widget"""
    def __init__(self,axSliderFrame,nFrames):
        self.nFrames = nFrames
        self.sFrame = Slider(axSliderFrame, 'Frame no.', 0,nFrames, valinit=0,valfmt='%d')
        self.frameClick = 0 # the position on the slider the user clicks
        self.frameRunning = 0 # how many frames on from the user's click we have moved
        def updateFrameSlider(val):
            self.frameClick = int(self.sFrame.val)
            self.frameRunning = 0
        self.sFrame.on_changed(updateFrameSlider)
    def getActualFrame(self):
        # the actual frame is where the user has clicked
        # plus how many times the loop has run
        return (self.frameClick+self.frameRunning)%self.nFrames
    def incrementFrame(self):
        # go round the loops
        actualFrame = self.getActualFrame()
        self.sFrame.vline.set_xdata([actualFrame,actualFrame])
        self.frameRunning+=1

# TODO: make this a template and do different ones for SS and CNN
class playBackClass(object):
    """a class for playing back a tub and showing the results of
    - image processing
    - various different pilots """
    def __init__(self):
        self.M = M
        self.nCrop = 50
    def setupPlotLineFollower(self,userAngles,img):
        # set up figure and subplots
        fig = plt.figure()
        ax1 = plt.subplot2grid((3,2),(0,0))
        ax2 = plt.subplot2grid((3,2),(0,1))
        ax3 = plt.subplot2grid((3,2),(1,0),colspan=2)
        # set up the axis for showing the user/pilot steering
        ax3.set_ylim([-90*np.pi/180,90*np.pi/180])
        ax3.plot(userAngles)
        pilotAnglePlot, = ax3.plot(self.pilotAngles,linestyle = '' ,marker = '.')
        currentPlot, =ax3.plot(0,userAngles[0],marker = 'o')
        self.nPlotLines = 11

        linesX = np.ones((2,self.nPlotLines))
        linesY = np.ones((2,self.nPlotLines))
        for x in range(0, self.nPlotLines):
            linesY[:,x] = linesY[:,x]*x
        edgeLines = ax2.plot(linesX,linesY,color = 'green')
        imPlot = ax1.imshow(img,animated=True)

        # get actual steering angle from the saved data
        steerLineTub, = ax1.plot([80,80],[0,80],color='orange')
        # get output steering angle from the driver
        steerLinePilot, = ax1.plot([80,80],[0,80],color='blue')
        imPlotProcessed = ax2.imshow(self.imProc(img),animated=True)
        plotObjects = {
        "pilotAnglePlot":pilotAnglePlot,
        "currentPlot":currentPlot,
        "edgeLines":edgeLines,
        "imPlot":imPlot,
        "imPlotProcessed":imPlotProcessed,
        "steerLineTub":steerLineTub,
        "steerLinePilot":steerLinePilot
        }
        return fig,plotObjects
    def updatePlotLineFollower(self,userAngle,pilotAngle,img,plotObjects):
        if pilotAngle is not None:
            plotObjects["steerLinePilot"].set_data([80,80+40*np.sin(pilotAngle)],[80,80-40*np.cos(pilotAngle)])
            self.pilotAngles[self.actualFrame] = pilotAngle
            plotObjects["pilotAnglePlot"].set_ydata(self.pilotAngles)
        lines = self.plf.lf.getAngle.lines
        if lines is None:
            nLines = 0
        else:
            nLines = len(lines)
        cntPlot = 0
        while cntPlot<self.nPlotLines:
            if cntPlot<nLines:
                for x1,y1,x2,y2 in lines[cntPlot]:
                    if (np.abs(y1-y2)>15):
                        plotObjects["edgeLines"][cntPlot].set_data([x1,x2],[y1,y2])
            else:
                plotObjects["edgeLines"][cntPlot].set_data([0,0],[0,0])
            cntPlot+=1
        plotObjects["steerLineTub"].set_data([80,80+40*np.sin(userAngle)],[80,80-40*np.cos(userAngle)])
        plotObjects["imPlot"].set_array(img)
        plotObjects["imPlotProcessed"].set_array(self.imProc(img))
        plotObjects["currentPlot"].set_data(self.actualFrame%self.nFrames, userAngle)
    def run(self,args, parser):

        if args.tub is None:
            print("ERR>> --tub argument missing.")
            parser.print_help()
            return
        self.doEdges = args.edge
        self.doTransform = args.transform
        self.pilot = pilotLF(self.doTransform)
        print('doEdges')
        print(self.doEdges)
        print('doTransform')
        print(self.doTransform)
        conf = os.path.expanduser(args.config)
        if not os.path.exists(conf):
            print("No config file at location: %s. Add --config to specify\
                 location or run from dir containing config.py." % conf)
            return

        self.cfg = dk.load_config(conf)
        print(self.cfg.DATA_PATH)
        self.tub = Tub(os.path.join(self.cfg.DATA_PATH,args.tub))
        userAngles = []
        self.nFrames = len(self.tub.get_index(shuffled=False))
        self.pilotAngles = np.ones(self.nFrames)*100
        for iRec in self.tub.get_index(shuffled=False):
            record = self.tub.get_record(iRec)
            userAngle = float(record["user/angle"])
            userAngle = (userAngle-0.5)*2*45*np.pi/180.
            userAngles.append(userAngle)
        self.index = self.tub.get_index(shuffled=False)
        record = self.tub.get_record(1)
        img = record["cam/image_array"]


        fig,plotObjects = self.setupPlotLineFollower(userAngles,img)
        # set up slider
        axSliderFrame = plt.axes([0.2, 0.08, 0.4, 0.03], facecolor='gray')
        sliderFrame = sliderPlayBack(axSliderFrame,self.nFrames)

        # set up steering lines

        def animate(i):
            # update the slider, which gives us the actual frame we're interested in
            self.actualFrame = sliderFrame.getActualFrame()
            sliderFrame.incrementFrame()
            record = self.tub.get_record(self.actualFrame+1) # tubs start at 1
            img = record["cam/image_array"]
            userAngle = float(record["user/angle"])
            # turn user angle into an actual angle
            userAngle = (userAngle-0.5)*2*45*np.pi/180.
            pilotAngle, pilotThrottle = self.pilot.run(img)
            angle = self.pilot.lc.lastAngleOut
            intercept = self.pilot.lc.lastIntercept
            self.updatePlotLineFollower(userAngle,angle,img,plotObjects)

            return img,
        ani = animation.FuncAnimation(fig, animate, np.arange(1, self.tub.get_num_records()),
                                  interval=100, blit=False)

        plt.show()
    def imProc(self,imgIn):
        # depending on options process image to show in top right frame
        # almost always crop to remove anything above track level
        if self.nCrop>=1:
            imgIn = imgIn[self.nCrop:,:,:]
        # extract edges
        if self.doEdges:
            imgIn =getEdges(imgIn)
        # perspective transform
        if self.doTransform:
            imgIn = cv2.warpPerspective(imgIn,M,(imgIn.shape[1],imgIn.shape[0]))
        imgOut = imgIn
        return imgOut
