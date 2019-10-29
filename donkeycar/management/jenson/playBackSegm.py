#!/usr/bin/env python3
"""
playback a tub and look at the output of the semantic segmentation
"""

import os
import sys

import numpy as np

from donkeycar.management.jenson.myPlayback import playBackClassLine
from donkeycar.parts.cv_pilot import pilotNull
from donkeycar.parts.jenson.segmPart import segmPart
class playBackClassSegm(playBackClassLine):
    def __init__(self,args):
        super().__init__(args)
        self.segmPart = segmPart()
    def getPilot(self,args):
        return pilotNull()
    def updatePlot(self,userAngle,pilotAngle,img,plotObjects):
        self.updatePlotProc(userAngle,pilotAngle,img,plotObjects)
    def imProc(self,imgIn):
        # run the semantic segmentation part here
        imOut = self.segmPart.run(imgIn)
        return imOut
