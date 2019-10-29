"""
Process an image using semantic segmentation to extract the track

"""

import os
import sys
from donkeycar.management.segm.model import Model_fromLoad


class segmPart():
    """
    semantic segmentation loading a pre-trained model in the
    donkey car part format
    """
    def __init__(self):
        # TODO: load model here- copy model pilot as much as poss
        pass
    def run(self,imgIn):
        # perform semantic segmentation here
        imgOut = imgIn
        return imgOut
    def run_threaded(self):
        pass
    def update(self):
        pass
