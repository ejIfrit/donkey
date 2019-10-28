#!/usr/bin/env python3
"""
playback a tub and show what the image procesisng is doing,
and what the outputs look linke
"""

import os
import sys
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import donkeycar as dk
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import numpy as np
#import parts
from donkeycar.parts.datastore import Tub, TubHandler, TubGroup
from donkeycar.parts.cv_pilot import pilotLF, pilotNull
from donkeycar.management.jensoncal import getEdges
from donkeycar import utils
print('imports done')
