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
import pickle as pkl
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from functools import partial
import multiprocessing as mp
from imutils import face_utils
from hyperparams import Hyperparams as hp
from face_alignment import FaceAlignment, LandmarksType
from utils import detector, predictor, preprocess_input

global face_alignment
face_alignment = FaceAlignment(LandmarksType._2D, device='cuda')

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

    frames = np.empty((n_frames, hp.img_size[0], hp.img_size[1], 3), np.dtype('uint8'))

    fn, ret = 0, True
    while fn < n_frames and ret:
        ret, img = cap.read()
        frames[fn] = cv2.cvtColor(cv2.resize(img,hp.img_size[:2]), cv2.COLOR_BGR2RGB)
        fn += 1

    cap.release()
    return frames

def select_random_frames(frames, K = hp.K):
    """
    Selects K+1 random frames from a list of frames.
    :param frames: Iterator of frames.
    :param frames: Iterator of frames.
    :return: List of selected frames.
    """
    S = []
    while len(S) <= K:
        s = random.randint(0, len(frames)-1)
        if s not in S:
            S.append(s)

    return [frames[s] for s in S]

def select_random_indices(length, K = hp.K):
    """
    Selects K+1 random indices from a list of iteration length.
    :param length: Iteration length.
    :return: List of selected indices.
    """
    S = []
    while len(S) <= K:
        s = random.randint(0, length-1)
        if s not in S:
            S.append(s)

    return S

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
    
    #landmarks = get_dlib(frame)
    landmarks = np.int32(face_alignment.get_landmarks_from_image(frame)[0])
    
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
    cv2.polylines(data,[landmarks[60:, :]],True,(127.5,0,255), 1)
    
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

def get_frame_data(frames, K = hp.K):
    K_plus_frames = select_random_frames(frames, K = hp.K)
        
    K_plus_ldmks = [plot_landmarks(f) for f in K_plus_frames]
           
    tx, ty = np.float32(K_plus_frames.pop()), np.float32(K_plus_ldmks.pop())
    x, y = np.float32(K_plus_frames), np.float32(K_plus_ldmks)
    
    tx = preprocess_input(tx, mode='tf')
    ty = preprocess_input(ty, mode='tf')
    x = preprocess_input(x, mode='tf')
    y = preprocess_input(y, mode='tf')
    
    tx = np.expand_dims(tx, axis=0) if len(tx.shape) != 4 else tx
    ty = np.expand_dims(ty, axis=0) if len(ty.shape) != 4 else ty
    
    x = np.expand_dims(x, axis=0) if len(x.shape) != 5 else x
    y = np.expand_dims(y, axis=0) if len(y.shape) != 5 else y
    
    return x, y, tx, ty

def preprocess():
    '''
    preprocess the dataset, extracts video frames, plots ldmk points and stores K+1 pairs as pickled data file.
    '''
    video_list = get_video_list(hp.dataset)
    os.makedirs(hp.preprocessed, exist_ok=True)
    for video in tqdm(video_list):
        folder, files = video
        vid_name = os.path.split(folder)[-1]
        if not '{}.pkl'.format(vid_name) in os.listdir(hp.preprocessed):
            if not  contains_only_videos(files):
                print('In {} All files are not video files.'.format(vid_name))
                continue        
            try:
                tot_frames = [] 
                for file in files:            
                    tot_frames.append(extract_frames(os.path.join(folder, file)))
                            
                tot_frames = np.concatenate(tuple(tot_frames),axis=0)
                frames = select_random_frames(tot_frames, K = hp.K)
                data = []
                for i in range(len(frames)):
                    x = frames[i]
                    y = plot_landmarks(x)
                    data.append({
                        'frame': x,
                        'landmarks': y,
                    })
                pkl.dump(data, open(os.path.join(hp.preprocessed, "{}.pkl".format(vid_name)), 'wb'))
            except:
                print('{} file can\'t be processed.'.format(vid_name))
                continue
        else:
            print('{}.pkl file already exist.'.format(vid_name))
            continue

def data(fine_tune = False, shuffle_frames = True):
    '''
    Iteration function for feeding input data to the Neural Network.

    Args:
        fine_tune (bool): Whether input to to fine tunning model or regular.
        shuffle_frames (bool): Whether to shuffle the K+1 frames.

    Returns:
        TensorFlow dataset iterator.
    '''
    
    files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(hp.preprocessed)
            for filename in files
            if filename.endswith('.pkl')
            ]
    files.sort()
    indexes = [idx for idx in range(len(files))]
        
    def generator():            
        for vid_id in indexes:            
            
            data = pkl.load(open(files[vid_id], 'rb'))
            
            if shuffle_frames:
                random.shuffle(data)

            K_plus_frames = []
            K_plus_ldmks = []
            for d in data:
                K_plus_frames.append(d['frame'])
                K_plus_ldmks.append(d['landmarks'])

            tx, ty = np.float32(K_plus_frames.pop()), np.float32(K_plus_ldmks.pop())
            x, y = np.float32(K_plus_frames), np.float32(K_plus_ldmks)
            
            tx = preprocess_input(tx, mode='tf')
            ty = preprocess_input(ty, mode='tf')
            x = preprocess_input(x, mode='tf')
            y = preprocess_input(y, mode='tf')
            
            yield np.int32(np.array(vid_id).reshape(-1)), np.float32(x), np.float32(y), np.float32(tx), np.float32(ty)
    
    output_types_  = (tf.int32,tf.float32,tf.float32,tf.float32,tf.float32)
    output_shapes_ = (tf.TensorShape([1,]),
                      tf.TensorShape([hp.K] + list(hp.img_size)),
                      tf.TensorShape([hp.K] + list(hp.img_size)),
                      tf.TensorShape(list(hp.img_size)),
                      tf.TensorShape(list(hp.img_size)))

    dataset = tf.data.Dataset.from_generator(generator,
                                       output_types= output_types_, 
                                       output_shapes= output_shapes_)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.repeat()
    dataset = dataset.batch(hp.batch)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    
    return next_batch

if __name__ == '__main__':
    preprocess()