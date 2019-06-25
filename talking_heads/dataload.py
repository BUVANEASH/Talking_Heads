#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:28:53 2019

@author: avantariml
"""
from __future__ import print_function

import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from imutils import face_utils
from utils import detector, predictor, preprocess_input
from hyperparams import Hyperparams as hp

def get_video_list(source = hp.dataset):
    """
    Extracts a list of paths to videos to pre-process during the current run.

    :param source: Path to the root directory of the dataset.
    :return: List of paths to videos.
    """
    video_list = []
    
    for root, dirs, files in tqdm(os.walk(source)):
        if len(files) > 0:
            assert contains_only_videos(files) and len(dirs) == 0
            video_list.append((root, files))

    return video_list

def contains_only_videos(files, extension='.mp4'):
    """
    Checks whether the files provided all end with the specified video extension.
    :param files: List of file names.
    :param extension: Extension that all files should have.
    :return: True if all files end with the given extension.
    """
    return len([x for x in files if os.path.splitext(x)[1] != extension]) == 0

def extract_frames(video):
    """
    Extracts all frames of a video file. Frames are extracted in BGR format, but converted to RGB. The shape of the
    extracted frames is [height, width, channels]. Be aware that PyTorch models expect the channels to be the first
    dimension.
    :param video: Path to a video file.
    :return: NumPy array of frames in RGB.
    """
    cap = cv2.VideoCapture(video)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = np.empty((n_frames, h, w, 3), np.dtype('uint8'))

    fn, ret = 0, True
    while fn < n_frames and ret:
        ret, img = cap.read()
        frames[fn] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fn += 1

    cap.release()
    return frames

def select_random_frames(frames):
    """
    Selects K+1 random frames from a list of frames.
    :param frames: Iterator of frames.
    :return: List of selected frames.
    """
    S = []
    while len(S) <= hp.K:
        s = random.randint(0, len(frames)-1)
        if s not in S:
            S.append(s)

    return [frames[s] for s in S]


def plot_landmarks(frame):
    """
    Creates an RGB image with the landmarks. The generated image will be of the same size as the frame where the face
    matching the landmarks.

    The image is created by plotting the coordinates of the landmarks using matplotlib, and then converting the
    plot to an image.

    Things to watch out for:
    * The figure where the landmarks will be plotted must have the same size as the image to create, but matplotlib
    only accepts the size in inches, so it must be converted to pixels using the DPI of the screen.
    * A white background is printed on the image (an array of ones) in order to keep the figure from being flipped.
    * The axis must be turned off and the subplot must be adjusted to remove the space where the axis would normally be.

    :param frame: Image with a face matching the landmarks.
    :return: RGB image with the landmarks.
    """
    data = np.ones_like(frame)*255
    
    landmarks = get_dlib(frame)
    
    # Head
    cv2.polylines(data,[landmarks[:17, :]],False,(0,255,0),2)        
    # Eyebrows
    cv2.polylines(data,[landmarks[17:22, :]],False,(255,127.5,0), 2) 
    cv2.polylines(data,[landmarks[22:27, :]],False,(255,127.5,0), 2)
    # Nose
    cv2.polylines(data,[landmarks[27:31, :]],False,(0,0,255), 2)
    cv2.polylines(data,[landmarks[31:36, :]],False,(0,0,255), 2)
    # Eyes
    cv2.polylines(data,[landmarks[36:42, :]],True,(255,0,0), 2)
    cv2.polylines(data,[landmarks[42:48, :]],True,(255,0,0), 2)
    # Mouth
    cv2.polylines(data,[landmarks[48:60, :]],True,(127.5,0,255), 2)
    #cv2.polylines(data,[landmarks[60:, :]],True,(127.5,0,255), 1)
    
    return np.array(data)

def get_dlib(image):
    '''
    Extracts dlib facial land mark points from the given image

    Args:
        image (ndarray `unint8`): Input image.

    Returns:
        dlib facial land mark points.
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #cv2.resize(image,(0,0),fx=1/2,fy=1/2)
    rects = detector(gray, 1)
    shape_pts = predictor(gray,rects[0])
    dlib_pts = face_utils.shape_to_np(shape_pts)
    return np.array(dlib_pts, dtype=np.int32)

def get_frame_data(frames):
    K_plus_frames = select_random_frames(frames)
        
    K_plus_ldmks = [plot_landmarks(f) for f in K_plus_frames]
           
    tx, ty = np.float32(K_plus_frames.pop()), np.float32(K_plus_ldmks.pop())
    x, y = np.float32(K_plus_frames), np.float32(K_plus_ldmks)
    
    tx = preprocess_input(tx, mode='tf')
    ty = preprocess_input(ty, mode='tf')
    x = preprocess_input(x, mode='tf')
    y = preprocess_input(y, mode='tf')
    
    tx = np.expand_dims(tx, axis=0) if len(tx.shape) != 4 else tx
    ty = np.expand_dims(ty, axis=0) if len(ty.shape) != 4 else ty
    
    x = np.expand_dims(x, axis=0) if len(x.shape) != 4 else x
    y = np.expand_dims(y, axis=0) if len(y.shape) != 4 else y
    
    return x, y, tx, ty
    

def get_video_data(video):
    
    folder, files = video

    try:
        assert contains_only_videos(files)
        frames = np.concatenate([extract_frames(os.path.join(folder, f)) for f in files])
        
        return get_frame_data(frames)  
        
    except:
        return [None, None, None, None]

def data(fine_tune = False):
    
    video_list = get_video_list(hp.dataset)
    idx = [i for i in range(0, len(video_list)-1)]
    random.shuffle(idx)
    
    with tf.device('/cpu:0'):
    
        def generator():            
            for vid_id in idx:            
                x, y, tx, ty = get_video_data(video_list[vid_id])
                if None in [x,y,tx,ty]:
                    continue
                yield np.int32(np.array(vid_id).reshape(-1)), np.float32(x), np.float32(y), np.float32(tx), np.float32(ty)
        
        output_types_  = (tf.int32,tf.float32,tf.float32,tf.float32,tf.float32)
        output_shapes_ = (tf.TensorShape([1,]),
                          tf.TensorShape([None] + list(hp.img_size)),
                          tf.TensorShape([None] + list(hp.img_size)),
                          tf.TensorShape([None] + list(hp.img_size)),
                          tf.TensorShape([None] + list(hp.img_size)))
    
        dataset = tf.data.Dataset.from_generator(generator,
                                           output_types= output_types_, 
                                           output_shapes= output_shapes_)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.repeat()
        dataset = dataset.batch(1)
        iterator = dataset.make_one_shot_iterator()
        next_batch = iterator.get_next()
    
    return next_batch