#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:04:06 2019

@author: avantariml
"""
import numpy as np
import tensorflow as tf
from blocks import embedder, generator, discriminator
from loss import loss_cnt, loss_adv, loss_mch, loss_dsc
from utils import learning_rate_decay
from hyperparams import Hyperparams as hp


class Graph():
    
    def __init__(self, mode="train", fine_tune = False, num_gpus=1):
        
        # Set flag
        self.training = True if mode=="train" else False
        self.fine_tune = fine_tune

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        
        if self.training:
            
            self.x = tf.placeholder(tf.float32, [None] + list(hp.img_size), name='x')
            self.y = tf.placeholder(tf.float32, [None] + list(hp.img_size), name='y')
            self.tx = tf.placeholder(tf.float32, [None] + list(hp.img_size), name='tx')
            self.ty = tf.placeholder(tf.float32, [None] + list(hp.img_size), name='ty')
            self.idx = tf.placeholder(tf.int32, [1,], name='idx')
            
            # Training Scheme
            self.learning_rate_EG = learning_rate_decay(hp.learning_rate_EG, self.global_step)
            self.optimizer_EG = tf.train.AdamOptimizer(learning_rate=self.learning_rate_EG)
            tf.summary.scalar("learning_rate_EG", self.learning_rate_EG)
            
           
            self.learning_rate_D = learning_rate_decay(hp.learning_rate_D, self.global_step)
            self.optimizer_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate_D)
            tf.summary.scalar("learning_rate_D", self.learning_rate_D)
            
            self.E = embedder(sn=True, is_training=self.training if not self.fine_tune else False, reuse=False)
            
            # Calculate average encoding vector for video
            self.e_hat, self.psi_hat = self.E(self.x,self.y)
            
            if not self.fine_tune:
                self.G = generator(sn=True, is_training=self.training, reuse=False, fine_tune=self.fine_tune, psi_hat_init = None)
                self.D, self.W = discriminator(sn=True, is_training=self.training, reuse=tf.AUTO_REUSE, fine_tune=self.fine_tune, e_new = None)
            else:
                self.G = generator(sn=True, is_training=self.training, reuse=False, fine_tune=self.fine_tune, psi_hat_init = self.psi_hat)
                self.D = discriminator(sn=True, is_training=self.training, reuse=tf.AUTO_REUSE, fine_tune=self.fine_tune, e_new = self.e_hat)
            
            # Generate frame using landmarks from frame t    
            self.x_hat = self.G(self.ty, self.psi_hat)
            
            # Generate real score for real and face (image,ldmk) pair
            self.r_x_hat, self.D_act_hat = self.D(self.x_hat, self.ty, self.idx)
            self.r_x, self.D_act =self.D(self.x, self.y, self.idx)
            
            self.loss_CNT = loss_cnt(self.tx, self.x_hat)
            self.loss_ADV = loss_adv(self.r_x_hat, self.D_act, self.D_act_hat)
            self.loss_MCH = loss_mch(self.e_hat, tf.reshape(tf.nn.embedding_lookup(self.W,self.idx), shape=[-1,1]))
            
            if not self.fine_tune:
                self.loss_EG =  self.loss_CNT + self.loss_ADV + self.loss_MCH 
            else:
                self.loss_EG =  self.loss_CNT + self.loss_ADV
            
            tf.summary.scalar("loss_CNT", self.loss_CNT)
            tf.summary.scalar("loss_ADV", self.loss_ADV)
            tf.summary.scalar("loss_MCH", self.loss_MCH)
            tf.summary.scalar("loss_EG", self.loss_EG)
            
            self.grads_EG = self.optimizer_EG.compute_gradients(self.loss_EG)
            
            ## gradient clipping
            #self.clipped_EG = [(tf.clip_by_value(grad, -1., 1.),var) for grad,var in self.grads_EG]           
            self.train_EG = self.optimizer_EG.apply_gradients(self.grads_EG, global_step=self.global_step)


            if not self.fine_tune:
                self.loss_D = loss_dsc(self.r_x, self.r_x_hat)
                tf.summary.scalar("loss_D", self.loss_D)
                
                self.grads_D = self.optimizer_D.compute_gradients(self.loss_D)
                
                ## gradient clipping
                #self.clipped_D = [(tf.clip_by_value(grad, -1., 1.),var) for grad,var in self.grads_D]
                self.train_D = self.optimizer_D.apply_gradients(self.grads_D, global_step=self.global_step)
            
            tf.summary.image('Generator/in/y', self.ty)
            tf.summary.image('Generator/out/x_hat', self.x_hat)
            tf.summary.image('Embedder/y', self.y[np.random.randint(0,hp.K+1)])
            tf.summary.image('Embedder/x', self.x[np.random.randint(0,hp.K+1)])
            
            # Summary
            self.merged = tf.summary.merge_all()
            
        else:
            
            self.x = tf.placeholder(tf.float32, [None] + list(hp.img_size), name='x')
            self.y = tf.placeholder(tf.float32, [None] + list(hp.img_size), name='y')
            self.ty = tf.placeholder(tf.float32, [None] + list(hp.img_size), name='tx')
            
            if not self.fine_tune:
                self.E = embedder(sn=True, is_training=self.training if not self.fine_tune else False, reuse=False)
                self.G = generator(sn=True, is_training=self.training, reuse=False, fine_tune=self.fine_tune, psi_hat_init = None)
            else:
                self.G = generator(sn=True, is_training=self.training, reuse=False, fine_tune=self.fine_tune, psi_hat_init = None)