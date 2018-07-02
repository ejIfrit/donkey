#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    test.py (playback) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--no_cache]
    test.py (annotate) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--no_cache]
    test.py (train) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--no_cache]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
    --js             Use physical joystick.
"""
from docopt import docopt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import sys
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import glob
import donkeycar as dk
from donkeycar.parts.datastore import Tub, TubHandler, TubGroup
from donkeycar import utils
from jensoncal import getEdges


class LineBuilder:
    def __init__(self, line, lineFit):
        self.line = line
        self.lineFit = lineFit
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        #self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.z  = []

#    def __call__(self, event):
#        print('click', event)
#        if event.inaxes!=self.line.axes: return
    def __call__(self,xdata,ydata):
        self.xs.append(xdata)
        self.ys.append(ydata)
        self.line.set_data(self.xs, self.ys)

        if len(self.xs)>3:
            self.bestFit()

        self.line.figure.canvas.draw()

    def bestFit(self):
        self.z = np.polyfit(self.ys, self.xs, 2)
        print('z:' + str(self.z))
        p = np.poly1d(self.z)
        self.lineFit.set_data(p(self.ys),self.ys)
    def clear(self):
        self.xs = []
        self.ys = []
        self.z = []
        self.line.set_data(self.xs, self.ys)
        self.lineFit.set_data([], [])
        self.line.figure.canvas.draw()
    def newLine(self,zIn,lChop = 45,hImg = 120):
        self.line.set_data([], [])
        p = np.poly1d(zIn)
        self.z = np.array(zIn)
        if len(self.ys)==0:
            tempy = np.linspace(lChop,hImg,5)
            self.lineFit.set_data(p(tempy),tempy-lChop)
        else:
            self.lineFit.set_data(p(self.ys),self.ys)
        self.line.figure.canvas.draw()


class LineBuilderTub:
    def __init__(self, tub_names,line, lineFit, axNext, axRedo,doPreProc = False):
        tubgroup = TubGroup(tub_names)
        self.lineBuilder  = LineBuilder(line, lineFit)
        tub_paths = utils.expand_path_arg(tub_names)
        tubs = [Tub(path) for path in tub_paths]
        self.tub = tubs[0]
        self.tubInds = self.tub.get_index(shuffled=False)
        self.kTub = 36
        self.tub.current_ix = self.tubInds[self.kTub]
        self.nRecords = len(self.tubInds)
        self.axNext = axNext
        self.axRedo = axRedo
        self.axFit = line.axes
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.doPreProc = doPreProc
        if doPreProc:
            self.lChop = 45
            self.lImage = 120
        else:
            self.lChop = 0
            self.lImage = 120
        record = self.tub.get_record(self.tubInds[self.kTub])
        img = record["cam/image_array"]
        bestfit = record["bestfit"]
        if self.doPreProc:
            print('yes to preprocessing')
            self.im = self.axFit.imshow(getEdges(img[self.lChop:,:,:]))
        else:
            print('no preprocessing')
            self.im = self.axFit.imshow(img)
        if "bestfit" in record:
            print("plot this line")
            self.lineBuilder.newLine(record["bestfit"],self.lChop,self.lImage)
        else:
            print('no best fit line')
        self.axFit.figure.canvas.draw()

    def __call__(self, event):
        print('clicknext', event)
        if event.inaxes==self.axNext:
            print('next')
            print(len(self.lineBuilder.z))
            if len(self.lineBuilder.z)>0:
                jsonthing = self.tub.get_json_record(self.tubInds[self.kTub])
                jsonthing["bestfit"] = self.lineBuilder.z.tolist()
                print(jsonthing)
                self.tub.write_json_record(jsonthing)
                self.kTub+=1
                self.kTub%=self.nRecords
                self.tub.current_ix = self.tubInds[self.kTub]
                record = self.tub.get_record(self.tubInds[self.kTub])
                if "bestfit" in record:
                    self.lineBuilder.newLine(record["bestfit"],self.lChop,self.lImage)
                else:
                    self.lineBuilder.clear()
                self.plotImg()
        if event.inaxes==self.axRedo:
            print('redo')
            self.lineBuilder.clear()
        if event.inaxes==self.axFit:
            print('fit')
            self.lineBuilder.__call__(event.xdata,event.ydata)
    def plotImg(self):
        record = self.tub.get_record(self.tubInds[self.kTub])
        img = record["cam/image_array"]
        if self.doPreProc:
            self.im.set_data(getEdges(img[self.lChop:,:,:]))
        else:
            self.im.set_data(img)
        self.axFit.set_title('Tub' + str(self.tubInds[self.kTub]))
        self.axFit.figure.canvas.draw()
    def setPreProc(self,doPreProc):
        self.doPreProc = True
    def checkTubs(self):
        nRecords = len(self.tubInds)
        print("nRecords = " + str(nRecords))
        isOK = True
        nBad = 0
        a = []
        b = []
        c = []
        for k in range(nRecords):
            record = self.tub.get_record(self.tubInds[k])

            if not "bestfit" in record:
                isOK = False
                nBad+=1
            else:
                a.append(record["bestfit"][0])
                b.append(record["bestfit"][1])
                c.append(record["bestfit"][2])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(a,b,c)
        return isOK,nBad








def playback(cfg,tub_names='/Users/edwardjackson/d_ej4/data/test/tub_21_18-05-01'):

    tubgroup = TubGroup(tub_names)
    tub_paths = utils.expand_path_arg(tub_names)
    tubs = [Tub(path) for path in tub_paths]
#    for tub in tubs:
#            num_records = tub.get_num_records()
#            print(num_records)
 #           for iRec in tub.get_index(shuffled=False):
    tub = tubs[0]

    tubInds = tub.get_index(shuffled=False)
    record = tub.get_record(tubInds[0])
    print('tub id = ' +str(tub.current_ix))
    tub.current_ix = tubInds[0]
    print('tub id = ' +str(tub.current_ix))
    jsonthing = tub.get_json_record(tubInds[0])
    print(jsonthing)
    sys.stdout.flush()
    img = record["cam/image_array"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click to add points')

    axNext = fig.add_axes([0.05,0.25,0.1,0.1])
    axNext.set_facecolor('xkcd:yellow green')
    axNext.set_yticklabels([])
    axNext.set_xticklabels([])
    axRedo = fig.add_axes([0.05,0.1,0.1,0.1])
    axRedo.set_facecolor('xkcd:rust')
    axRedo.set_yticklabels([])
    axRedo.set_xticklabels([])
    line, = ax.plot([], [], linestyle="none", marker="o", color="b")
    lineFit, = ax.plot([], [], color="orange")
    im = ax.imshow(img)
    linebuilder = LineBuilder(line, lineFit)
    plt.draw()

    jsonthing["bestfit"] = linebuilder.z.tolist()
    print(jsonthing)
    sys.stdout.flush()
    tub.write_json_record(jsonthing)


def annotate(cfg,tub_names='/Users/edwardjackson/d_ej4/data/test/tub_21_18-05-01',doPreProc = False):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click to add points')

    axNext = fig.add_axes([0.05,0.25,0.1,0.1])
    axNext.set_facecolor('xkcd:yellow green')
    axNext.set_yticklabels([])
    axNext.set_xticklabels([])
    axRedo = fig.add_axes([0.05,0.1,0.1,0.1])
    axRedo.set_facecolor('xkcd:rust')
    axRedo.set_yticklabels([])
    axRedo.set_xticklabels([])
    line, = ax.plot([], [], linestyle="none", marker="o", color="b")
    lineFit, = ax.plot([], [], color="orange")
    linebuilder = LineBuilderTub(tub_names,line, lineFit,axNext, axRedo, doPreProc = doPreProc)
    plt.show(block=False)
    return linebuilder

def trainOnLines(cfg,tub_names='/Users/edwardjackson/d_ej4/data/test/tub_21_18-05-01',doPreProc = False):
    lb = annotate(cfg,tub_names,doPreProc)
    print("Hi Ed")
    isOK,nBad = lb.checkTubs()
    if isOK:
        print("Tub Good!")
    else:
        print("Not enough lines")
        print(str(nBad)+" empty records")

    plt.show()


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    print(args)
    if args['playback']:
        tub = args['--tub']
        cache = not args['--no_cache']
        playback(cfg, tub)
    if args['annotate']:
        tub = args['--tub']
        cache = not args['--no_cache']
        annotate(cfg, tub)
    if args['train']:
        tub = args['--tub']
        cache = not args['--no_cache']
        trainOnLines(cfg, tub)
