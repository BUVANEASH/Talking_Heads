#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:05:08 2019

@author: BUVANEASH
"""
import os
import argparse
import tensorflow as tf
from KGAN import KGAN
from dataload import extract_frames
from hyperparams import Hyperparams as hp

def _str_to_bool(s):
  if s.lower() not in ['true', 'false']:
      raise ValueError('Need bool; got %r' % s)
  return s.lower() == 'true'

def add_boolean_argument(parser, name, default=False):
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
      '--' + name, nargs='?', default=default, const=True, type=_str_to_bool)
  group.add_argument('--no' + name, dest=name, action='store_false')
  
def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-v','--vid', type=str, default=None,
                       help='input video for fine tunning')
    parser.add_argument('-l','--log', type=str, default=None,
                       help='log path')
    add_boolean_argument(parser, "fine", default=False)
    
    args = parser.parse_args()
    
    if args.fine:
        print("Fine Tuning...")
        
    if args.fine and type(args.vid) == type(None):
            raise ValueError('Need path to input video')
    
    if type(args.vid) != type(None):
        if args.vid.split('.')[-1] != 'mp4':
            raise ValueError('Need input video file; got {}'.format(args.vid))
    
    config = tf.ConfigProto()
    config.gpu_options.allocator_type="BFC"
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    
    sess = tf.InteractiveSession(config=config)
    
    kgan  = KGAN(sess = sess, mode="train",fine_tune = args.fine)
    
    kgan.update(hp.__dict__)
        
    kgan.frames = extract_frames(args.vid) if args.fine else None
    
    if args.fine and not type(args.log) == type(None):
        os.makedirs(args.log, exist_ok=True)
        kgan.fine_logdir = args.log
    elif not type(args.log) == type(None):
        os.makedirs(args.log, exist_ok=True)
        kgan.logdir = args.log
        
    kgan.build()
    
    kgan.train()
    
    
if __name__ == '__main__':
    main()
    
    
    