import os
import numpy as np
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'

from donkey.management.jensoncal import lookAtFloorImg2
from donkey.management.jensoncal import getEdges

import donkeycar as dk

def clip( x,  myMin,  myMax):
    if (x < myMin): return myMin
    if (x > myMax): return myMax
    return x

def getAngle(imgIn):    
    edgesThenTransform = lookAtFloorImg2(getEdges(imgIn),balance = 1.0)[120:,:]
    minLineLength = 30
    maxLineGap = 10
    lines = cv2.HoughLinesP(edgesThenTransform,1,np.pi/180,40,minLineLength,maxLineGap)
    myX = []
    myY = []
    myX1 = []
    myIntercept = []
    myGrad = []
    for x in range(0, (len(lines))):
        for x1,y1,x2,y2 in lines[x]:
            if (np.abs(y1-y2)>10) and nPlot<len(edgeLines):
                        myX.append(x2-x1)
                        myY.append(y1-y2)
                        myIntercept.append((x2*(y1-90)-x1*(y2-90))/(y1-y2))                        
                        myGrad.append((x2-x1)/(y1-y2))
                        nPlot+=1
            
            #np.arctan2(y, x) * 180 / np.pi
            myAngle = np.arctan(np.median(myGrad))
            interceptOut = np.median(myIntercept)
    return myAngle, interceptOut 

class lineFollower():
    def __init__(self, *args, **kwargs):
        
        
    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        myAngle, interceptOut = getAngle(img_arr)
        #angle_binned, throttle = self.model.predict(img_arr)
        #print('throttle', throttle)
        #angle_certainty = max(angle_binned[0])
        print('angle', myAngle)
        print('interceptOut', interceptOut)
        throttle = 0.5
        angle_unbinned = clip(myAngle,-1.,1.)
        return angle_unbinned, throttle
        
    def shutdown(self):
        pass
