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

import os
import sys
from docopt import docopt

import donkeycar as dk
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
#import parts
from donkeycar.parts.camera import PiCamera
from donkeycar.parts.transform import Lambda

from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.parts.datastore import Tub, TubHandler, TubGroup
from donkeycar.parts.controller import LocalWebController, JoystickController
from donkeycar import utils
from jensoncal import lookAtFloorImg
print('imports done')

def playback(cfg, tub_names, model_name=None):
    if not tub_names:
        tub_names = os.path.join(cfg.DATA_PATH, '*')
    tubgroup = TubGroup(tub_names)
    
    if not model_name is None:
        from donkeycar.parts.keras import KerasCategorical
        kl = KerasCategorical()
        kl.load(model_name)
        pilot_angles = []
        pilot_throttles  = []
        
    
    
    print('tub_names', tub_names)
    
    tub_paths = utils.expand_path_arg(tub_names)
    print('TubGroup:tubpaths:', tub_paths)
    user_angles = []
    user_throttles  = []
    
        
    tubs = [Tub(path) for path in tub_paths]
    for tub in tubs:
            num_records = tub.get_num_records()
            print(num_records)
            for iRec in tub.get_index(shuffled=False):
                record = tub.get_record(iRec)
            #record = tubs.get_record(random.randint(1,num_records+1))
                img = record["cam/image_array"]
                user_angle = float(record["user/angle"])
                user_throttle = float(record["user/throttle"])
                user_angles.append(user_angle)
                user_throttles.append(user_throttle)
                if not model_name is None:
                    pilot_angle, pilot_throttle = kl.run(img)
                    pilot_angles.append(pilot_angle)
                    pilot_throttles.append(pilot_throttle)
            
    
    record = tubs[0].get_record(random.randint(1,num_records+1))
    user_angle = float(record["user/angle"])
    user_throttle = float(record["user/throttle"])
    print(img.shape)
    print('-----')
    print(user_angle)
    print(user_throttle)
    plt.figure()
    plt.imshow(img)
    plt.plot([80,80+10*user_throttle*np.cos(user_angle)],[120,120+100*user_throttle*np.sin(user_angle)])
    
    plt.figure()
    plt.plot(user_angles)
    plt.plot(user_throttles)
    
    
    
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((2,2),(0,0))
    record = tubs[0].get_record(1)
    img = record["cam/image_array"]
    imPlot = ax1.imshow(img,animated=True)
    
    
    floorImg = lookAtFloorImg(img,maxHeight = 0)
    #print(floorImg)
    ax3 = plt.subplot2grid((2,2),(0,1))
    imPlot2 = ax3.imshow(floorImg[100: , :, :],animated=True)
    
    ax2 = plt.subplot2grid((2,2),(1,0), colspan=2)
    line1, = ax2.plot(user_angles)
    line2, = ax2.plot(user_throttles)
    if not model_name is None:
        line4, = ax2.plot(pilot_angles)
        line5, = ax2.plot(pilot_throttles)
    line3, = ax2.plot([0,0],[-1,1])
    
    
    
    def animate(i):
        record = tubs[0].get_record(i)
        img = record["cam/image_array"]
        imPlot.set_array(img)
        imPlot2.set_array(lookAtFloorImg(img)[100: , :, :])
        line3.set_data([i,i],[-1,1])
        #print(i)
        #sys.stdout.flush()
        return imPlot,


    # Init only required for blitting to give a clean slate.
    def init():
        record = tubs[0].get_record(1)
        img = record["cam/image_array"]
        imPlot.set_array(img)
        line3.set_data([0,0],[-1,1])
        return imPlot,
    
    
    
    ani = animation.FuncAnimation(fig, animate, np.arange(1, tubs[0].get_num_records()),
                              interval=100, blit=False)
                              
    plt.show()
    
    #user_angles.append(user_angle)
    #user_throttles.append(user_throttle)
    #pilot_angles.append(pilot_angle)
    #pilot_throttles.append(pilot_throttle)
            

    

if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    print(args)
    if args['playback']:
        tub = args['--tub']
        model = args['--model']
        cache = not args['--no_cache']
        playback(cfg, tub, model)


