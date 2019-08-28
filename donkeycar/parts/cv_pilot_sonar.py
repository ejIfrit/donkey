import os
import numpy as np
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import donkeycar as dk

from donkeycar.parts.cv_pilot import clip, lineController


class lineControllerSonar(lineController):
    def __init__(self, cutoff=40,*args, **kwargs):
        lineController.__init__(self, *args, **kwargs)
        self.cutoff=cutoff
    def run(self,angleIn,interceptIn,distIn):
        if angleIn is not None:
            angle_intercept1 = (interceptIn-self.dIntercept)*self.kIntercept
            angle_imAngle2 = (angleIn-self.dAngle)*self.kAngle
            angle_unbinned = angle_intercept1*0.+angle_imAngle2*1.0
            angle_unbinned = clip(angle_unbinned,-1.,1.)
            self.lastIntercept = interceptIn
            self.lastAngleOut =angle_unbinned
        else:
            angle_unbinned = self.lastAngleOut*0.5
            self.lastAngleOut =angle_unbinned
        throttle = 1.0
        if distIn<self.cutoff:
            throttle = 0.
        print('throttle'+str(throttle))
        return angle_unbinned, throttle
    def shutdown(self):
        pass
