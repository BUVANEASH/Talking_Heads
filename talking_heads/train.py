#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:05:08 2019

@author: avantariml
"""

import tensorflow as tf
from KGAN import KGAN
from hyperparams import Hyperparams as hp

def main():
    
    config = tf.ConfigProto()
    config.gpu_options.allocator_type="BFC"
    #config.allow_soft_placement = True
    #config.log_device_placement = False
    config.gpu_options.allow_growth = True
    
    sess = tf.Session(config=config)
    
    kgan  = KGAN(sess = sess)
    
    kgan.update(hp.__dict__)
    
    kgan.build()
    
    kgan.train()
    
    
if __name__ == '__main__':
    main()
    
    
    