#!/usr/bin/env python
'''
Predict Server
Create a server to accept image inputs and run them against a trained neural network.
This then sends the steering output back to the client.
Author: Tawn Kramer
'''
from __future__ import print_function
import os
import argparse
import sys
import numpy as np
import h5py
import json
#from keras.models import load_model
#import matplotlib.pyplot as plt
import time
import asyncore
import json
import socket
from PIL import Image
import pygame
import conf
import predict_server_EJ
from donkeycar.management.threshProc import getThreshRGB, getPolyFit
pygame.init()
ch, row, col = conf.ch, conf.row, conf.col

size = (col*2, row*2)
pygame.display.set_caption("sdsandbox data monitor")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
camera_surface = pygame.surface.Surface((col,row),0,24).convert()
myfont = pygame.font.SysFont("monospace", 15)

def screen_print(x, y, msg, screen):
    label = myfont.render(msg, 1, (255,255,0))
    screen.blit(label, (x, y))

def display_img(img, steering):
    #print('got an image to display')
    pygame.event.get()
    #print(img)

    img = img.swapaxes(0, 1)
    #print(img.shape)
    img = getThreshRGB(img)
    coeffs = getPolyFit(img)
    img = np.array([img, img,img])
    img = img.swapaxes(0, 2)
    img = img.swapaxes(0, 1)

    # draw frame
    #print('after')
    #print(img)
    #print(img.shape)
    #print('draw a frame')
    pygame.surfarray.blit_array(camera_surface, img)
    camera_surface_2x = pygame.transform.scale2x(camera_surface)
    screen.blit(camera_surface_2x, (0,0))
    #steering value
    #print('add steering value')
    if len(coeffs)==3:
        screen_print(10, 10, 'ax^2: ' + str(coeffs[2]) + ' bx: ' + str(coeffs[1]) + ' c: ' + str(coeffs[0]), screen)
    else:
        screen_print(10, 10, 'no fit', screen)
    #print('flip and update')
    pygame.display.flip()
    pygame.display.update()

# ***** main loop *****
if __name__ == "__main__":
  #parser = argparse.ArgumentParser(description='prediction server with monitor')
  #parser.add_argument('model', type=str, help='model name. no json or keras.')
  #args = parser.parse_args()

  address = ('0.0.0.0', 9090)

  try:
    predict_server_EJ.run_steering_server_nomodel(address,image_folder=None, image_cb=display_img)
  except KeyboardInterrupt:
    print('got ctrl+c break')
