import os
import numpy as np
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import donkeycar as dk
from donkeycar.management.jensoncal import lookAtFloorImg2
from donkeycar.management.jensoncal import getEdges
import time
import argparse

# perspective transformation from camera to birds eye view
M=np.array([[0.7288447030443173, 5.383427898971357, 9.457197002209224], [-5.551115123125783e-16, 2.8193645731219714, 3.8191672047105385e-14], [-0.0017493890126173033, 0.06997556050469123, 1.0]])


def clip( x,  myMin,  myMax):
    if (x < myMin): return myMin
    if (x > myMax): return myMax
    return x

class getAngleClass():
    def __init__(self,doTransform = False,nCrop = 50,minLineLength = 10, maxLineGap = 20, nLinesMax = 11,printDebug = False):
        self.doTransform = doTransform
        self.nCrop = nCrop
        self.minLineLength = minLineLength
        self.maxLineGap = maxLineGap
        self.nLinesMax = nLinesMax
        self.printDebug = printDebug
        self.lines = None
    def run(self,imgIn):

        edgeImg =getEdges(imgIn[self.nCrop:,:,:])
        lines = cv2.HoughLinesP(edgeImg,1,np.pi/180,25,
            np.array([]),self.minLineLength,self.maxLineGap)

        cnt = 0
        myAngle=None
        interceptOut = None
        myIntercept = []
        myGrad = []
        if lines is not None:
            if self.printDebug:
                print("----lines----")
                print(len(lines))
            for x in range(0, (len(lines))):
                for x1,y1,x2,y2 in lines[x]:
                    if (np.abs(y1-y2)>10) and cnt<self.nLinesMax:
                        if self.doTransform:
                            original = np.array([((x1, y1), (x2, y2))], dtype=np.float32)
                            converted = cv2.perspectiveTransform(original, M)
                            x1 = converted[0][0][0]
                            x2 = converted[0][1][0]
                            y1 = converted[0][0][1]
                            y2 = converted[0][1][1]
                            lines[x] = (x1,y1,x2,y2)
                        myIntercept.append((x2*(y1-90)-x1*(y2-90))/(y1-y2))
                        myGrad.append((x2-x1)/(y1-y2))
                        cnt+=1

            #np.arctan2(y, x) * 180 / np.pi
            myAngle = np.arctan(np.median(myGrad))
            if np.isnan(myAngle): myAngle = None
            interceptOut = np.median(myIntercept)
            if np.isnan(interceptOut): interceptOut = None
        else:
            myAngle = None
            interceptOut = None
        self.lines = lines
        return myAngle, interceptOut


class lineFollower():
    def __init__(self,doTransform = False,printDebug = False):
        self.imIn = []
        self.angle_unbinned = 0.
        self.throttle = 0.
        self.tempInt = 0
        self.doTransform = doTransform
        self.printDebug = printDebug
        self.getAngle = getAngleClass(doTransform = doTransform,printDebug = printDebug)
    def run(self, img_arr):

        t0 = time.time()
        #if self.doTransform:
        #    myAngle, interceptOut = getAngleTransform(img_arr,self.printDebug)
        #else:
        #    myAngle, interceptOut = getAngle(img_arr,self.printDebug)
        myAngle, interceptOut = self.getAngle.run(img_arr)
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
        # the function runs in its own thread
        while True:
            self.angle_unbinned,self.throttle  = self.run(self.imIn)

    def run_threaded(self,img_arr):
        self.imIn = img_arr
        return self.angle_unbinned,self.throttle
    def shutdown(self):
        pass

class lineController():
    def __init__(self):
        self.dAngle = 0. # the d is for demand
        self.dIntercept = 70.
        self.kIntercept = 1.
        self.kAngle = 1.
        self.lastAngleOut = 0.
        self.lastSteeringAngleOut = 0.
        self.lastIntercept = 70.
    def run(self,angleIn,interceptIn):
        if angleIn is not None:
            angle_intercept1 = (interceptIn-self.dIntercept)*self.kIntercept
            angle_imAngle2 = (angleIn-self.dAngle)*self.kAngle
            steer_angle_unbinned = angle_intercept1*0.+angle_imAngle2*1.0
            steer_angle_unbinned = clip(steer_angle_unbinned,-1.,1.)
            self.lastIntercept = interceptIn
            self.lastSteeringAngleOut =steer_angle_unbinned
            self.lastAngleOut =angleIn
        else:
            steer_angle_unbinned = self.lastSteeringAngleOut
        throttle = 0.
        return steer_angle_unbinned, throttle # slightly confusing as
                                        # there are 2 angles, the actual angle and the angle if the lines
    def shutdown(self):
        pass
class pilotNull():
    # a pilot that doesn't actually do anything, but needed for cases
    # when we either are just looking at
    def __init__(self):
        self.imIn = None
        self.angle_unbinned = 0.5
        self.throttle = 0
    def run(self,imIn):
        angle_unbinned = 0.5
        throttle = 0
        return angle_unbinned,throttle
    def update(self):
        # the function runs in its own thread
        while True:
            time.sleep(0.1)
    def run_threaded(self,img_arr):
        self.imIn = img_arr
        return self.angle_unbinned,self.throttle
class pilotLF():
    def __init__(self,doTransform = False,printDebug = False):
        self.lf = lineFollower(doTransform,printDebug)
        self.lc = lineController()
        self.imIn = None
        self.angle_unbinned = 0.5
        self.throttle = 0
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
