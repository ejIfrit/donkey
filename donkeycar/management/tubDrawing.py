#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it. 

Usage:
    test.py (playback) [--tub=<tub1,tub2,..tubn>]  [--model=<model>] [--no_cache]
    
Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
    --js             Use physical joystick.
"""
from docopt import docopt
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import glob
import donkeycar as dk
from donkeycar.parts.datastore import Tub, TubHandler, TubGroup
from donkeycar import utils


class LineBuilder:
    def __init__(self, line, lineFit):
        self.line = line
        self.lineFit = lineFit
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        
        if len(self.xs)>3:
            self.bestFit()
        
        self.line.figure.canvas.draw()
        
        
    def bestFit(self):
        z = np.polyfit(self.ys, self.xs, 2)
        print('z:' + str(z))
        p = np.poly1d(z)
        self.lineFit.set_data(p(self.ys),self.ys)
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
    img = record["cam/image_array"]



    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click to add points')
    line, = ax.plot([], [], linestyle="none", marker="o", color="b")
    lineFit, = ax.plot([], [], color="orange")
    im = ax.imshow(img)
    linebuilder = LineBuilder(line, lineFit)
    plt.show()


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    print(args)
    if args['playback']:
        tub = args['--tub']
        cache = not args['--no_cache']
        playback(cfg, tub)