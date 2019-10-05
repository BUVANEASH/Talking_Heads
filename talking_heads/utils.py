#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:28:49 2019

@author: BUVANEASH
"""
import os
import dlib
import numpy as np
from PIL import Image
import subprocess
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

def ffmpeg_images2video(frames, video_out, audio_in = None, fps = 29.97):
    '''
    Method function to generate video from images and audio using ffmpeg.

    Args:
        frames : List of frames images to compose video.
        video_out (str): Path to save video.
        audio_in (str): Input audio file path.
        fps (float): Video frames per second.
    '''
    h,w,_ = frames[0].shape
    
    if audio_in:
        command = ['ffmpeg',
                    '-s', '{}x{}'.format(w,h), # size of one frame
                    '-pix_fmt', 'rgb24',
                    '-r',str(fps), # frames per second
                    '-f','image2pipe',#'-an', # Tells FFMPEG not to expect any audio/media/new_hdd/Dataset/Mouth_Syn/output/a10_297_rms.wa
                    '-i','-',
                    '-i',audio_in,
                    '-c:v','copy',
                    '-map','0:v:0',
                    '-map','1:a:0',
                    '-c:a','aac',
                    '-b:a','192k',
                    '-strict','-2',
                    '-vcodec','mpeg4',
                    '-qscale','0',
                    video_out,
                    '-y', # (optional) overwrite output file if it exists
                    ]
    else:
        command = ['ffmpeg',
                    '-s', '{}x{}'.format(w,h), # size of one frame
                    '-pix_fmt', 'rgb24',
                    '-r',str(fps), # frames per second
                    '-f','image2pipe',#'-an', # Tells FFMPEG not to expect any audio/media/new_hdd/Dataset/Mouth_Syn/output/a10_297_rms.wa
                    '-i','-',
                    '-an',
                    '-vcodec','mpeg4',
                    '-qscale','0',
                    video_out,
                    '-y', # (optional) overwrite output file if it exists
                    ]

    print('RUNNING ',' '.join(command))
    
    pipe = subprocess.Popen( command, stdin=subprocess.PIPE)

    for ss in frames:
        im = Image.fromarray(np.uint8(ss))
        im.save(pipe.stdin, 'JPEG')

    pipe.stdin.close()
    pipe.wait()