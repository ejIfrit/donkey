import os
import numpy as np
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import donkeycar as dk
from donkeycar.management.jensoncal import lookAtFloorImg2
from donkeycar.management.jensoncal import getEdges
import time


# perspective transformation from camera to birds eye view
M=np.array([[0.7288447030443173, 5.383427898971357, 9.457197002209224], [-5.551115123125783e-16, 2.8193645731219714, 3.8191672047105385e-14], [-0.0017493890126173033, 0.06997556050469123, 1.0]])


def clip( x,  myMin,  myMax):
    if (x < myMin): return myMin
    if (x > myMax): return myMax
    return x
# TODO: why are these even separate functions?

class getAngleClass():
    def __init__():
        pass
    def run(args,parser):
        doTransform = args.doTransform
        nCrop = args.nCrop
        minLineLength = args.minLineLength
        maxLineGap = args.maxLineGap
        nLinesMax = args.nLinesMax
        printDebug = args.printDebug
        edgeImg =getEdges(imgIn[50:,:,:])
        lines = cv2.HoughLinesP(edgeImg,1,np.pi/180,25,np.array([]),minLineLength,maxLineGap)
        nPlot = 0
        myAngle=None
        interceptOut = None
        if lines is not None:
            if printDebug:
                print("----lines----")
                print(len(lines))
            for x in range(0, (len(lines))):
                for x1,y1,x2,y2 in lines[x]:
                    if (np.abs(y1-y2)>10) and nPlot<nLinesMax:
                        if doTransform:
                            original = np.array([((x1, y1), (x2, y2))], dtype=np.float32)
                            converted = cv2.perspectiveTransform(original, M)
                            x1 = converted[0][0][0]
                            x2 = converted[0][1][0]
                            y1 = converted[0][0][1]
                            y2 = converted[0][1][1]
                        myIntercept.append((x2*(y1-90)-x1*(y2-90))/(y1-y2))
                        myGrad.append((x2-x1)/(y1-y2))
                        nPlot+=1

            #np.arctan2(y, x) * 180 / np.pi
            myAngle = np.arctan(np.median(myGrad))
            if np.isnan(myAngle): myAngle = None
            interceptOut = np.median(myIntercept)
            if np.isnan(interceptOut): interceptOut = None
        else:
            myAngle = None
            interceptOut = None
        return myAngle, interceptOut

def getAngleTransform(imgIn,printDebug = False):
    if len(imgIn)>0:
        #edgesThenTransform = lookAtFloorImg2(getEdges(imgIn),balance = 1.0)[120:,:]
        edgesThenTransform =getEdges(imgIn[50:,:,:])
        minLineLength = 10
        maxLineGap = 20
        lines = cv2.HoughLinesP(edgesThenTransform,1,np.pi/180,25,np.array([]),minLineLength,maxLineGap)
        myX = []
        myY = []
        myX1 = []
        myIntercept = []
        myGrad = []
        nPlot = 0
        nLinesMax = 11
        myAngle=None
        interceptOut = None
        if lines is not None:
            if printDebug:
                print("----lines----")
                print(len(lines))
            for x in range(0, (len(lines))):
                for x1,y1,x2,y2 in lines[x]:
                    if (np.abs(y1-y2)>10) and nPlot<nLinesMax:
                        #myX.append(x2-x1)
                        #myY.append(y1-y2)
                        original = np.array([((x1, y1), (x2, y2))], dtype=np.float32)
                        converted = cv2.perspectiveTransform(original, M)
                            #edgeLines2[nPlot].set_data([converted[0][0][0],converted[0][1][0]],[converted[0][0][1],converted[0][1][1]])
                        newx1 = converted[0][0][0]
                        newx2 = converted[0][1][0]
                        newy1 = converted[0][0][1]
                        newy2 = converted[0][1][1]
                        myIntercept.append((newx2*(newy1-90)-newx1*(newy2-90))/(newy1-newy2))
                        myGrad.append((newx2-newx1)/(newy1-newy2))

                        #myIntercept.append((x2*(y1-90)-x1*(y2-90))/(y1-y2))
                        #myGrad.append((x2-x1)/(y1-y2))
                        nPlot+=1

            myAngle = np.arctan(np.median(myGrad))
            if np.isnan(myAngle): myAngle = None
            interceptOut = np.median(myIntercept)
            if np.isnan(interceptOut): interceptOut = None
    else:
        myAngle = None
        interceptOut = None
    return myAngle, interceptOut


def getAngle(imgIn,printDebug = False,doTransform = False,nCrop = 50):
    if len(imgIn)>0:
        #edgesThenTransform = lookAtFloorImg2(getEdges(imgIn),balance = 1.0)[120:,:]
        edgesThenTransform =getEdges(imgIn[nCrop:,:,:])
        minLineLength = 10
        maxLineGap = 20
        lines = cv2.HoughLinesP(edgesThenTransform,1,np.pi/180,25,np.array([]),minLineLength,maxLineGap)
        myX = []
        myY = []
        myX1 = []
        myIntercept = []
        myGrad = []
        nPlot = 0
        nLinesMax = 11
        myAngle=None
        interceptOut = None
        if lines is not None:
            if printDebug:
                print("----lines----")
                print(len(lines))
            for x in range(0, (len(lines))):
                for x1,y1,x2,y2 in lines[x]:
                    if (np.abs(y1-y2)>10) and nPlot<nLinesMax:
                        myX.append(x2-x1)
                        myY.append(y1-y2)
                        myIntercept.append((x2*(y1-90)-x1*(y2-90))/(y1-y2))
                        myGrad.append((x2-x1)/(y1-y2))
                        nPlot+=1
            myAngle = np.arctan(np.median(myGrad))
            if np.isnan(myAngle): myAngle = None
            interceptOut = np.median(myIntercept)
            if np.isnan(interceptOut): interceptOut = None
    else:
        myAngle = None
        interceptOut = None
    return myAngle, interceptOut

class lineFollower():
    def __init__(self,doTransform = False,printDebug = False):
        self.imIn = []
        self.angle_unbinned = 0.
        self.throttle = 0.
        self.tempInt = 0
        self.doTransform = doTransform
        self.printDebug = printDebug

    def run(self, img_arr):
        #img_arr = img_arr.reshape((1,) + img_arr.shape)
        t0 = time.time()
        if self.doTransform:
            myAngle, interceptOut = getAngleTransform(img_arr,self.printDebug)
        else:
            myAngle, interceptOut = getAngle(img_arr,self.printDebug)
        t1 = time.time()
        total = t1-t0
        if self.printDebug:
            print('that took')
            print(total)
        self.tempInt +=1
        if self.printDebug:
            print('count', self.tempInt)
            print('angle', myAngle)
            print('interceptOut', interceptOut)

        return myAngle, interceptOut

    def update(self):
        # the funtion runs in its own thread
        while True:
            self.angle_unbinned,self.throttle  = self.run(self.imIn)

    def run_threaded(self,img_arr):
        self.imIn = img_arr
        return self.angle_unbinned,self.throttle
    def shutdown(self):
        pass

class lineController():
    def __init__(self, *args, **kwargs):
        self.dAngle = 0. # the d is for demand
        self.dIntercept = 70.
        self.kIntercept = 1.
        self.kAngle = 20.
        self.lastAngleOut = 0.
        self.lastIntercept = 70.
    def run(self,angleIn,interceptIn):
        if angleIn is not None:
            angle_intercept1 = (interceptIn-self.dIntercept)*self.kIntercept
            angle_imAngle2 = (angleIn-self.dAngle)*self.kAngle
            angle_unbinned = angle_intercept1*0.+angle_imAngle2*1.0
            angle_unbinned = clip(angle_unbinned,-1.,1.)
            self.lastIntercept = interceptIn
            self.lastAngleOut =angle_unbinned
        else:
            angle_unbinned = self.lastAngleOut
        throttle = 0.
        return angle_unbinned, throttle
    def shutdown(self):
        pass

class pilotLF():
    def __init__(self,doTransform = False,printDebug = False):
        self.lf = lineFollower(doTransform,printDebug)
        self.lc = lineController()
        self.imIn = None
    def run(self, imIn):
        angle, intercept = self.lf.run(imIn)
        angle_unbinned,throttle = self.lc.run(angle,intercept)
        # underscore because I'm matching output from dk pilot here
        return angle_unbinned,throttle
    def update(self):
        # the function runs in its own thread
        while True:
            if self.imIn is not None:
                self.angle_unbinned,self.throttle  = self.run(self.imIn)
            else:
                time.sleep(0.1)
    def run_threaded(self,img_arr):
        self.imIn = img_arr
        return self.angle_unbinned,self.throttle
