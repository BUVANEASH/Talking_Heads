#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:28:49 2019

@author: avantariml
"""
import os
import dlib
import tensorflow as tf

def find_files(name):
    '''
    Finds the file location from the parent directory

    Args:
        name: Filename.

    Returns:
        The absolute path to the file.

    Notes:
        Files present within the parent directory will only be returned.
    '''
    path = os.path.split(os.path.dirname(os.path.realpath('__file__')))[0]
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(find_files('shape_predictor_68_face_landmarks.dat'))

preprocess_input = tf.keras.applications.vgg19.preprocess_input

def learning_rate_decay(init_lr, global_step, warmup_steps = 4000.0): 
    #learning rate going to change dynamically which is one of the  core idea of 
    '''Noam scheme from tensor2tensor'''                             # Adam Optimization algo
    step = tf.to_float(global_step + 1)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
