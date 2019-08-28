#!/usr/bin/env python3
"""
based on https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py
"""

import sys
import numpy as np
import math
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import randint

from matplotlib import animation

class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0,v = 1.):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.predelta = None
        self.v = v

class donkeyCarSim:
    """
    two-wheeled follower class
    """

    def __init__(self,wb = 0.18):
        self.state = State()
        self.wb = wb # wheelbase (m)
        self.MAX_STEER = 25.*np.pi/180.
        self.dt = 0.1
        self.MAX_SPEED = 2.0
        self.MIN_SPEED = 0.0
        self.state.v = 1.
    def update(self):

        self.state = self.updatePredict(self.state,steerAngle)

    def updatePredict(self,state = State(),delta = 0.,a = 0.):
            stateOut = State()
            DT = self.dt
            WB = self.wb
            # input check
            if delta >= self.MAX_STEER:
                delta = self.MAX_STEER
            elif delta <= -self.MAX_STEER:
                delta = -self.MAX_STEER

            stateOut.x = state.x + state.v * math.cos(state.yaw) * DT
            stateOut.y = state.y + state.v * math.sin(state.yaw) * DT
            stateOut.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
            stateOut.v = state.v + a * DT

            if stateOut.v > self.MAX_SPEED:
                stateOut.v = self.MAX_SPEED
            elif stateOut.v < self.MIN_SPEED:
                stateOut.v = self.MIN_SPEED

            return stateOut

    def getPaths(self,nPaths = 5,nSim = 10):
            deltas = np.linspace(-self.MAX_STEER,self.MAX_STEER,nPaths)
            stateArraysOut = []
            for delta in deltas:
                stateTemp = State()
                stateArray = []
                k=0
                while k<nSim:
                    stateTemp = self.updatePredict(stateTemp,delta)
                    stateArray.append(stateTemp)
                    k+=1
                stateArraysOut.append(stateArray)
            return stateArraysOut
    def gen_dists(self,statesIn):
        nx = 160
        ny = 120
        wx_m = 1.
        wy_m = 1.
        xState = []
        yState = []
        imOut = np.zeros(shape=(ny,nx))
        for state in statesIn:
            xState.append(state.y)
            yState.append(state.x)
        xState = np.array(xState)
        yState = np.array(yState)
        xIn = np.linspace(-wx_m/2.,wx_m/2.,nx)
        yIn = np.linspace(0.,wy_m,ny)

        for ii in range(len(xIn)):
            x = xIn[ii]
            for jj in range(len(yIn)):
                y = yIn[jj]
                minDist = np.min(np.square(x-xState)+np.square(y-yState))
                imOut[ny-jj-1][ii] = minDist
        return imOut
    def gen_track(self):
        nx = 160
        ny = 120
        wx_m = 1.
        wy_m = 1.
        nBlocks = 5
        #xStart = randint(0, 5)
        xStart = 2
        imOut = np.zeros(shape=(ny,nx))

        imOut[40:120,int(xStart*nx/nBlocks):int((xStart+1)*nx/nBlocks-1)] = 1
        xStart = 1.5
        imOut[int(xStart*nx/nBlocks):int((xStart+1)*nx/nBlocks-1),0:80] = 1
        return imOut




# ***** main loop *****
if __name__ == "__main__":
    print("path planning simulation")
    dk = donkeyCarSim()
    nPaths = 5
    statesOut = dk.getPaths(nPaths)
    fig, axIms = plt.subplots(1, nPaths)
    print(len(statesOut))
    imTrack = dk.gen_track()
    #fig, axs = plt.subplots(1, 1)
    for states,axIm in zip(statesOut,axIms):
        x = []
        y = []
        for state in states:
            x.append(state.x)
            y.append(state.y)
        #print(x)
        #print(y)
        #axs.plot(y,x)

        imOut = dk.gen_dists(states)
        axIm.imshow(imOut,cmap='gray',extent=[-0.5,0.5,0,1])
        axIm.plot(y,x)
        axIm.set_xlim(-1,1)
        axIm.set_ylim(0,1)
        axIm.imshow(imTrack,extent=[-0.5,0.5,0,1],alpha=0.5)
        print('----min---')
        print(np.min(imOut[np.nonzero(imTrack)]))
        print('----sum---')
        print(np.sum(imOut[np.nonzero(imTrack)]))

    plt.show()
