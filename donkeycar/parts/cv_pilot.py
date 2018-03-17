import os
import numpy as np
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import donkeycar as dk
from donkeycar.management.jensoncal import lookAtFloorImg2
from donkeycar.management.jensoncal import getEdges
import time



def clip( x,  myMin,  myMax):
    if (x < myMin): return myMin
    if (x > myMax): return myMax
    return x

def getAngle(imgIn):  
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
        myAngle=0
        interceptOut = 0
        if lines is not None:
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
            
            #np.arctan2(y, x) * 180 / np.pi
            myAngle = np.arctan(np.median(myGrad))
            if np.isnan(myAngle): myAngle = -10
            interceptOut = np.median(myIntercept)
            if np.isnan(interceptOut): interceptOut = -10
    else:
        myAngle = -10
        interceptOut = -10        
    return myAngle, interceptOut 

class lineFollower():
    def __init__(self, *args, **kwargs):
        self.imIn = []
        self.angle_unbinned = 0.
        self.throttle = 0.
        self.tempInt = 0
        
        
    def run(self, img_arr):
        #img_arr = img_arr.reshape((1,) + img_arr.shape)
        t0 = time.time()
        myAngle, interceptOut = getAngle(img_arr)
        t1 = time.time()
        total = t1-t0
        print('that took')
        print(total)
        #angle_binned, throttle = self.model.predict(img_arr)
        #print('throttle', throttle)
        #angle_certainty = max(angle_binned[0])
        self.tempInt +=1
        print('count', self.tempInt)
        print('angle', myAngle)
        print('interceptOut', interceptOut)
        throttle = 0.
        if myAngle==-10:
             angle_unbinned=self.angle_unbinned
        else:
             #angle_unbinned = clip(angle_unbinned*5.,-1.,1.)
             angle_unbinned = clip((interceptOut-70)/4.,-1.,1.) 
        return angle_unbinned, throttle
        
    def update(self):
        #the funtion run in it's own thread
        while True:
            self.angle_unbinned,self.throttle  = self.run(self.imIn)
            
    def run_threaded(self,img_arr):
        self.imIn = img_arr
        return self.angle_unbinned,self.throttle
    def shutdown(self):
        pass
