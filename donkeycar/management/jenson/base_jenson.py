''' just as base.py has a list of functions that can be called
    with the donkey command, this script contains a list of functions
    that are activated when the user types $ donkey jenson ...'''
import sys
import argparse
import donkeycar as dk
from donkeycar.utils import *
from donkeycar.management.jenson.myPlayback import *
class JensonFuncs(object):
    """
    This is the class linked to the "donkey jenson" terminal command.
    """
    def run(self,args):

        commands = {
                'playback': playBackShell,
                    }


        if len(args) > 1 and args[0] in commands.keys():
            command = commands[args[0]]
            c = command()
            c.run(args[1:])
        else:
            dk.utils.eprint('Usage: The available commands using donkey jenson are:')
            dk.utils.eprint(list(commands.keys()))

class playBackShell(object):
    """
    Shell class for replaying a specific tub. Most of the heavy lifting is done inside
    the playBackClass itself
    """
    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='playback')
        parser.add_argument('--tub', help='The tub to play back from')
        parser.add_argument('--model', default='./models/drive.h5', help='path to model for NN controller')
        parser.add_argument('--line', action="store_true", help='use the line follower controller')
        parser.add_argument('--edge', action="store_true", help='extract the edges from the image')
        parser.add_argument('--transform', action="store_true", help='perspective transform the image')
        parser.add_argument('--car', action="store_true", help='perform image augmentation')
        parser.add_argument('--single', default=None, help='the model type to load')
        parser.add_argument('--config', default='~/d3_1/config.py', help='location of config file to use. default: ./config.py')
        #TODO: let me change where I get my data from just by changeing the name of the car
        parsed_args = parser.parse_args(args)
        return parsed_args, parser
    def playBackFactory(self,args):
        '''
        decide which version of playback we are using
        '''
        if args.line:
            # use the line follower
            return(playBackClassLine(args))
        else:
            # no pilot specified, just play back tubs
            return(playBackClassBase(args))

    def run(self, args):
        '''
        Make the playback class and get it to do its thing
        '''
        #TODO, at the moment, everything is just 'args'. Need to make This
        # a bit clearer- run just contains file arguments, playbackfactory sets up class
        args, parser = self.parse_args(args)
        print(args)
        pb = self.playBackFactory(args)
        pb.run(args,parser)
        #from donkeycar.management.makemovie import MakeMovie

        #mm = MakeMovie()
        #mm.run(args, parser)
