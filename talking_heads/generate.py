#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:40:11 2019

@author: BUVANEASH
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from KGAN import KGAN
from dataload import extract_frames, get_frame_data, get_input_frame_data, denormalize
from hyperparams import Hyperparams as hp
from utils import ffmpeg_images2video

def _str_to_bool(s):
  if s.lower() not in ['true', 'false']:
      raise ValueError('Need bool; got %r' % s)
  return s.lower() == 'true'

def add_boolean_argument(parser, name, default=False):
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
      '--' + name, nargs='?', default=default, const=True, type=_str_to_bool)
  group.add_argument('--no' + name, dest=name, action='store_false')


'''
args.fine = False

#args.fine = True

args.input = '/media/new_hdd1/Face_Morp_2.0/Talking_Heads/data/AK.mp4'

#args.input = '/media/new_hdd1/VoxCeleb-2/mp4/id05124/6hUV8ejPW8E/00103.mp4'

args.log = '/media/new_hdd1/Face_Morp_2.0/Talking_Heads/results/model/meta'

args.output = '/media/new_hdd1/Face_Morp_2.0/Talking_Heads/results/output'

args.combine = True

'''


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i','--input', type=str, default=None,
                       help='input video')
    parser.add_argument('-o','--output', type=str, default='./',
                       help='output path')
    parser.add_argument('-l','--log', type=str, default=None,
                       help='checkpint log dir')
    parser.add_argument('-k','--k', type=int, default=None,
                       help='K random frames for meta trained model')
    add_boolean_argument(parser, "fine", default=False)
    add_boolean_argument(parser, "combine", default=False)
    
    args = parser.parse_args()
    
    if args.fine and type(args.log) == type(None):
            raise ValueError('Need log path for fine tuned model')
            
    if type(args.input) != type(None):
        if args.input.split('.')[-1] != 'mp4':
            raise ValueError('Need input video file; got {}'.format(args.vid))
    
    if not args.fine:
        x, y, _, _ = get_frame_data(extract_frames(args.input), K = args.k if args.k else hp.K)
    else:
        x = None
        y = None
        
    tx, ty = get_input_frame_data(extract_frames(args.input))
    
    config = tf.ConfigProto()
    config.gpu_options.allocator_type="BFC"
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    
    sess = tf.InteractiveSession(config=config)
    
    kgan  = KGAN(sess = sess, mode="infer",fine_tune = args.fine)
    
    kgan.update(hp.__dict__)
           
    kgan.build()
    
    if args.fine and type(args.log) != type(None):
        kgan.fine_logdir = args.log
    elif type(args.log) != type(None):
        kgan.logdir = args.log
        
    frames = kgan.inference(ty = ty, x = x, y = y, batch = False)
    
    if type(args.output) != type(None):
        if os.path.isdir(args.output):
            args.output = os.path.join(args.output,'output.mp4')
    
    if args.combine:
        frames = np.concatenate((tx,ty,frames),axis=-2)
    
    frames = denormalize(frames)
    
    ffmpeg_images2video(frames, args.output, audio_in = None, fps = 29.97)

if __name__ == '__main__':
    main()