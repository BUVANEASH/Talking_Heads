#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:25:52 2019

@author: avantariml
"""
import os
import datetime
import numpy as np
import random
import pickle
import tensorflow as tf
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_tf_input
from utils import learning_rate_decay, preprocess_input
from network_ops import weight_init, weight_regularizer
from network_ops import resblock_down, resblock_down_no_instance_norm, resblock, resblock_condition, resblock_up_condition, self_attention
from network_ops import fully_connected, global_sum_pooling, relu, tanh, sigmoid
from dataload import get_video_list, get_frame_data

class KGAN():
    
    def __init__(self, sess, mode="train", fine_tune = False, model = "fine", frames = None):
        
        #Tensorflow session
        self.sess = sess
        
        # Set flag
        self.training = True if mode=="train" else False
        self.fine_tune = fine_tune
        self.frames = frames

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
#        # Dataset
#        self.dataset = "/opt/ml/input/data/VoxCeleb-2/Video/dev/mp4"
#        self.data = "/opt/ml/input/data"
#        self.preprocessed = os.path.join(self.data,"preprocessed")
#        
#        # logdir
#        self.model = model
#        self.modeldir = "/opt/ml/model/"
#        self.logdir = os.path.join(self.modeldir, "meta")
#        self.fine_logdir = os.path.join(self.modeldir, self.model)
               
        # No of training videos
#        self.train_videos = os.listdir(self.preprocessed)
        
        # Network Architecture parameters
        # Encoder channels and self-attention channel
        self.enc_down_ch = [64,128,256,512]
        self.enc_self_att_ch = 256
        
        # Decoder channels and self-attention channel
        self.dec_down_ch = [256,128,64,3]
        self.dec_self_att_ch = 256
        
        # Residual Block channel
        self.res_blk_ch = 512
        
        # Embedding Vector
        self.N_Vec = 512
        
        # Considering input and output channel in a residual block, multiple of 2 because beta and gamma affine parameter.
        self.split_lens = [self.res_blk_ch]*11*2 + \
                            [self.res_blk_ch]*2*2 + \
                            [self.res_blk_ch]*2*2 + \
                            [self.dec_down_ch[0]]*2*2 + \
                            [self.dec_down_ch[1]]*2*2 + \
                            [self.dec_down_ch[2]]*2*2 + \
                            [self.dec_down_ch[3]]*2
        
        # Activation outputs from VGGFace and VGG19
        self.vggface_feat_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']
        self.vgg19_feat_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
        
        # Training hyperparameters
        # Image Size
        self.img_size = (256, 256, 3)
        
        # K-shot learning,
        self.K = 8
        
        # batch size
        self.batch = 8

        # Loss weights
        self.loss_vgg19_wt =  1e-2
        self.loss_vggface_wt = 2e-3
        self.loss_fm_wt = 1e1
        self.loss_mch_wt = 8e1
        self.learning_rate_EG = 5e-5
        self.learning_rate_D = 2e-4
        self.num_iterations = 10000000
        
        # Logging
        self.log_step = 10
        self.save_step = 1000
        self.summary_step = 100
        
    def Embedder(self, x, y, sn=True, reuse=False):
	    
        with tf.variable_scope("embedder", reuse=reuse):

            xs = tf.reshape(x, [-1] + list(self.img_size)) # out (B*K)*256*256*3
            ys = tf.reshape(y, [-1] + list(self.img_size)) # out (B*K)*256*256*3
            
            xr = tf.image.resize_image_with_crop_or_pad(xs, 256, 256) # out (B*K)*256*256*3
            yr = tf.image.resize_image_with_crop_or_pad(ys, 256, 256) # out (B*K)*256*256*3
            
            c = tf.concat([xr,yr], axis=-1) # out 256*256*6
            
            e_rd1 = resblock_down_no_instance_norm(c, channels=self.enc_down_ch[0], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_1') # out (B*K)*128*128*64
            e_rd2 = resblock_down_no_instance_norm(e_rd1, channels=self.enc_down_ch[1], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_2') # out (B*K)*64*64*128
            e_rd3 = resblock_down_no_instance_norm(e_rd2, channels=self.enc_down_ch[2], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_3') # out (B*K)*32*32*256
            e_s1 = self_attention(e_rd3, channels=self.enc_self_att_ch, sn=sn, scope='self_attention') # out 32*32*256
            e_rd4 = resblock_down_no_instance_norm(e_s1, channels=self.enc_down_ch[3], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_4') # out (B*K)*16*16*512
            e_rd5 = resblock_down_no_instance_norm(e_rd4, channels=self.enc_down_ch[3], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_5') # out (B*K)*8*8*512
            e_rd6 = resblock_down_no_instance_norm(e_rd5, channels=self.enc_down_ch[3], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_6') # out (B*K)*4*4*512
            
            egs = global_sum_pooling(e_rd6) # out (B*K)*512
            er = relu(egs) # out (B*K)*512

            es = tf.reshape(er, [-1,self.K,self.N_Vec]) # out B*K*512

            e =  tf.reduce_mean(es,axis=1) # out B*512
            
            psi_hat = fully_connected(e, units=sum(self.split_lens),
                                      use_bias=False, is_training=self.training, 
                                      sn=False, scope='Projection_matrix') # out B*13062
            
            return tf.squeeze(e), tf.squeeze(psi_hat)
            
    def Generator(self,y, psi_Pe=None, psi_hat_init = None, sn=True, reuse=False):        
            
        def img2img(y, z):
            with tf.variable_scope("image-to-image", reuse=reuse):
                
                z_splits = tf.split(z, num_or_size_splits=self.split_lens)
                
                yr = tf.image.resize_image_with_crop_or_pad(y, 256, 256) # out B*256*256*3
                x_rd1 = resblock_down(yr, channels=self.enc_down_ch[0], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_1') # out B*128*128*64
                x_rd2 = resblock_down(x_rd1, channels=self.enc_down_ch[1], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_2') # out B*64*64*128
                x_rd3 = resblock_down(x_rd2, channels=self.enc_down_ch[2], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_3') # out B*32*32*256
                x_s1 = self_attention(x_rd3, channels=self.enc_self_att_ch, sn=sn, scope='self_attention_down') # out B*32*32*256
                x_rd4 = resblock_down(x_s1, channels=self.enc_down_ch[3], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_4') # out B*16*16*512
                x_rd5 = resblock_down(x_rd4, channels=self.enc_down_ch[3], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_5') # out B*8*8*512
                x_rd6 = resblock_down(x_rd5, channels=self.enc_down_ch[3], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_6') # out B*4*4*512
                
                x_r1 = resblock_condition(x_rd6, z_splits[:4], channels=self.res_blk_ch, use_bias=False, is_training=self.training, sn=sn, scope='resblock_1') # out B**4*512
                x_r2 = resblock_condition(x_r1, z_splits[4:8], channels=self.res_blk_ch, use_bias=False, is_training=self.training, sn=sn, scope='resblock_2') # out B*4*4*512
                x_r3 = resblock_condition(x_r2, z_splits[8:12], channels=self.res_blk_ch, use_bias=False, is_training=self.training, sn=sn, scope='resblock_3') # out B*4*4*512
                x_r4 = resblock_condition(x_r3, z_splits[12:16], channels=self.res_blk_ch, use_bias=False, is_training=self.training, sn=sn, scope='resblock_4') # out B*4*4*512
                x_r5 = resblock_condition(x_r4, z_splits[16:20], channels=self.res_blk_ch, use_bias=False, is_training=self.training, sn=sn, scope='resblock_5') # out B*4*4*512
                
                x_ru1 = resblock_up_condition(x_r5, z_splits[20:24], channels=self.res_blk_ch, use_bias=False, is_training=self.training, sn=sn, scope='resblock_up_1') # out B*8*8*512
                x_ru2 = resblock_up_condition(x_ru1, z_splits[24:28], channels=self.res_blk_ch, use_bias=False, is_training=self.training, sn=sn, scope='resblock_up_2') # out B*16*16*512
                x_ru3 = resblock_up_condition(x_ru2, z_splits[28:32], channels=self.dec_down_ch[0], use_bias=False, is_training=self.training, sn=sn, scope='resblock_up_3') # out B*32*32*256
                x_s2 = self_attention(x_ru3, channels=self.dec_self_att_ch, sn=sn, scope='self_attention_up') # out B*32*32*256
                x_ru4 = resblock_up_condition(x_s2, z_splits[32:36], channels=self.dec_down_ch[1], use_bias=False, is_training=self.training, sn=sn, scope='resblock_up_4') # out B*64*64*128
                x_ru5 = resblock_up_condition(x_ru4, z_splits[36:40], channels=self.dec_down_ch[2], use_bias=False, is_training=self.training, sn=sn, scope='resblock_up_5') # out B*128*128*64
                x_ru6 = resblock_up_condition(x_ru5, z_splits[40:44], channels=self.dec_down_ch[3], use_bias=False, is_training=self.training, sn=sn, scope='resblock_up_6') # out B*256*256*3
                
                x = tanh(x_ru6) # out B*256*256*3
                
                return x
            
        
        with tf.variable_scope("generator", reuse=reuse):
            
            psi_hat = tf.get_variable("AdaIN_params", shape=[sum(self.split_lens)], initializer=psi_hat_init if (self.training and self.fine_tune) else weight_init, 
                                    regularizer=weight_regularizer, 
                                    trainable=True if (self.training and self.fine_tune) else False)

            if not self.fine_tune:
                Pe = tf.assign(psi_hat, psi_Pe)
                with tf.control_dependencies([Pe]):
                    x = img2img(y,psi_hat)
                    return x
            else:
                x = img2img(y,psi_hat)
                return x
    
    def Discriminator(self, x, y, i=None, e_new = None, sn=True, reuse=False):
    
        def Discriminator_Net(x,y):
            with tf.variable_scope("discriminator_ConvNet", reuse=reuse):
                
                xr = tf.image.resize_image_with_crop_or_pad(x, 256, 256) # out B*256*256*3
                yr = tf.image.resize_image_with_crop_or_pad(y, 256, 256) # out B*256*256*3
                c = tf.concat([xr,yr], axis=-1) # out B*256*256*6
                
                o0 = resblock_down_no_instance_norm(c, channels=self.enc_down_ch[0], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_1') # out B*128*128*64
                o1 = resblock_down_no_instance_norm(o0, channels=self.enc_down_ch[1], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_2') # out B*64*64*128
                o2 = resblock_down_no_instance_norm(o1, channels=self.enc_down_ch[2], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_3') # out B*32*32*256
                o3 = self_attention(o2, channels=self.enc_self_att_ch, sn=sn, scope='self_attention') # out B*32*32*256
                o4 = resblock_down_no_instance_norm(o3, channels=self.enc_down_ch[3], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_4') # out B*16*16*512
                o5 = resblock_down_no_instance_norm(o4, channels=self.enc_down_ch[3], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_5') # out B*8*8*512
                o6 = resblock_down_no_instance_norm(o5, channels=self.enc_down_ch[3], use_bias=False, is_training=self.training, sn=sn, scope='resblock_down_6') # out B*4*4*512
                            
                orb = resblock(o6, channels=self.N_Vec, kernel=4, pads=[2,1], use_bias=False, is_training=self.training, sn=sn, scope='resblock') # out B*4*4*512
                
                ogs = global_sum_pooling(orb) # out B*1*512
                o = relu(ogs) # out B*1*512
                
                return tf.squeeze(o), [o0,o1,o2,o3,o4,o5,o6]
                       
        with tf.variable_scope("discriminator", reuse=reuse):
            
            self.W = tf.get_variable("W", shape=[self.train_videos,self.N_Vec], initializer=weight_init, regularizer=weight_regularizer)
            w_0 = tf.get_variable("w_0", shape=[1,self.N_Vec], initializer=weight_init, regularizer=weight_regularizer)
            
            w_hat = tf.get_variable("w_hat", shape=[1,self.N_Vec], initializer=tf.math.add(tf.reshape(e_new,shape=[-1,self.N_Vec]), w_0) if (self.training and self.fine_tune) else weight_init, 
                                    regularizer=weight_regularizer, 
                                    trainable=True if (self.training and self.fine_tune) else False)
    
            b = tf.get_variable("bias", shape=[1], initializer=weight_init, regularizer=weight_regularizer)
            
            o , d_act = Discriminator_Net(x,y)
            
            if not self.fine_tune:
                w_hat = tf.math.add(tf.nn.embedding_lookup(self.W,i), w_0)
                with tf.control_dependencies([w_hat]):
                    r = tf.squeeze(tf.matmul(tf.expand_dims(o,axis=1), tf.expand_dims(w_hat,axis=-1)))
                    r = tanh(tf.math.add(r, b))
                    return r, d_act
            else:
                r = tf.squeeze(tf.matmul(tf.expand_dims(o,axis=1), tf.expand_dims(w_hat,axis=-1)))
                r = tanh(tf.math.add(r, b))
                return r, d_act
    
    def VGGFACE(self, input_tensor=None, input_shape = (224, 224, 3)):

        vggface = VGGFace(input_tensor=input_tensor, model='vgg16', include_top=False, input_shape=input_shape)    
        outputs = []    
        for l in self.vggface_feat_layers:
            outputs.append(vggface.get_layer(l).output)        
        model = tf.keras.Model(inputs = vggface.input, outputs = outputs)    
        
        return model

    def VGG19(self, input_tensor=None, input_shape = (224, 224, 3)):
    
        vgg19 = tf.keras.applications.VGG19(input_tensor=input_tensor, include_top=False, input_shape=input_shape)    
        outputs = []    
        for l in self.vgg19_feat_layers:
            outputs.append(vgg19.get_layer(l).output)        
        model = tf.keras.Model(inputs = vgg19.input, outputs = outputs)
        
        return model
   
    def loss_cnt(self, xi, x_hat):
        
        vggface_xi = self.VGGFACE(input_tensor=xi, input_shape = self.img_size) # preprocess_tf_input(tf.multiply(tf.add(xi,1.0),127.5))
        vggface_x_hat = self.VGGFACE(input_tensor=x_hat, input_shape = self.img_size) # preprocess_tf_input(tf.multiply(tf.add(x_hat,1.0),127.5))
        
        vgg19_xi = self.VGG19(input_tensor=xi, input_shape = self.img_size)
        vgg19_x_hat = self.VGG19(input_tensor=x_hat, input_shape = self.img_size)
        
        vggface_loss = 0
        for i in range(len(vggface_xi.output)):
            vggface_loss  += tf.reduce_mean(tf.abs(vggface_xi.output[i] - vggface_x_hat.output[i]))
            
        vgg19_loss = 0
        for i in range(len(vgg19_xi.output)):
            vgg19_loss  += tf.reduce_mean(tf.abs(vgg19_xi.output[i] - vgg19_x_hat.output[i]))
    
        return vgg19_loss * self.loss_vgg19_wt + vggface_loss * self.loss_vggface_wt
          
    def loss_fm(self, d_act, d_act_hat):
        loss = 0
        for i in range(0, len(d_act)):
            loss += tf.reduce_mean(tf.abs(d_act[i] - d_act_hat[i]))
        return loss * self.loss_fm_wt
      
    def loss_adv(self, r_x_hat, d_act, d_act_hat):
        return -tf.reduce_mean(r_x_hat) + self.loss_fm(d_act, d_act_hat)
            
    def loss_mch(self, e_hat, W_i):
        return tf.reduce_mean(tf.abs(W_i - e_hat)) * self.loss_mch_wt
         
    def loss_dsc(self, r_x, r_x_hat):
        return tf.reduce_mean(tf.math.maximum(0.0, (1 + r_x_hat)) + tf.math.maximum(0.0, (1 - r_x)))
    
    
    def build(self):
        
        if self.training:
            # Training Scheme
            self.learning_rate_EG = learning_rate_decay(self.learning_rate_EG, self.global_step)
            self.optimizer_EG = tf.train.AdamOptimizer(learning_rate=self.learning_rate_EG)
            tf.summary.scalar("learning_rate_EG", self.learning_rate_EG)
            
           
            self.learning_rate_D = learning_rate_decay(self.learning_rate_D, self.global_step)
            self.optimizer_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate_D)
            tf.summary.scalar("learning_rate_D", self.learning_rate_D)
            
            
                
            self.x = tf.placeholder(tf.float32, [None] + list(self.img_size), name='x')
            self.y = tf.placeholder(tf.float32, [None] + list(self.img_size), name='y')
            self.tx = tf.placeholder(tf.float32, [None] + list(self.img_size), name='tx')
            self.ty = tf.placeholder(tf.float32, [None] + list(self.img_size), name='ty')
            if not self.fine_tune:
                self.idx = tf.placeholder(tf.int32, [1,], name='idx')
                
            # Embedder
            # Calculate average encoding vector for video and AdaIn params input
            self.e_hat, self.psi_hat = self.Embedder(self.x, self.y, sn=True, reuse=False)
            
            if not self.fine_tune:
                # Generator
                # Generate frame using landmarks from frame t    
                self.x_hat = self.Generator(self.ty, psi_Pe=self.psi_hat, sn=True, reuse=False)           
                # Discriminator
                # real score for fake image
                self.r_x_hat, self.D_act_hat = self.Discriminator(self.x_hat, self.ty, i=self.idx, e_new = None, sn=True, reuse=False)
                # real score for real image
                self.r_x, self.D_act =self.Discriminator(self.tx, self.ty, i=self.idx, e_new = None, sn=True, reuse=True)
            else:              
                x, y, _, _ = get_frame_data(self.frames)
                
                embedder_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'embedder')
                
                embedder_saver = tf.train.Saver(var_list=embedder_var_list)
                embedder_saver.restore(self.sess, tf.train.latest_checkpoint(self.logdir))
                
                self.sess.run(tf.global_variables_initializer)
                
                e_hat, psi_hat = self.sess.run([self.e_hat, self.psi_hat],
                                               feeddict = {self.x:x,
                                                           self.y:y})
                
                # Generator
                # Generate frame using landmarks from frame t    
                self.x_hat = self.Generator(self.ty, psi_Pe=None, psi_hat_init = psi_hat, sn=True, reuse=False)           
                # Discriminator
                # real score for fake image
                self.r_x_hat, self.D_act_hat = self.Discriminator(self.x_hat, self.ty, i=None, e_new = e_hat, sn=True, reuse=False)
                # real score for real image
                self.r_x, self.D_act =self.Discriminator(self.tx, self.ty, i=None, e_new = e_hat, sn=True, reuse=True)
            
            self.loss_CNT = self.loss_cnt(self.tx, self.x_hat)
            self.loss_ADV = self.loss_adv(self.r_x_hat, self.D_act, self.D_act_hat)
                
            if not self.fine_tune:
                self.loss_MCH = self.loss_mch(self.e_hat, tf.reshape(tf.nn.embedding_lookup(self.W,self.idx), shape=[-1,1]))
                
                self.loss_EG =  self.loss_CNT + self.loss_ADV + self.loss_MCH 
            else:
                self.loss_EG =  self.loss_CNT + self.loss_ADV
            
            self.loss_DSC = self.loss_dsc(self.r_x,self.r_x_hat)
            
            tf.summary.scalar("loss_CNT", self.loss_CNT)
            tf.summary.scalar("loss_ADV", self.loss_ADV)
            if not self.fine_tune:
                tf.summary.scalar("loss_MCH", self.loss_MCH)
            tf.summary.scalar("loss_EG", self.loss_EG)
            tf.summary.scalar("loss_DSC", self.loss_DSC)
            tf.summary.scalar("loss_r_x_hat", self.r_x_hat)
            tf.summary.scalar("loss_r_x", self.r_x)
            
            # Embedder & Generator Optimization
            EG_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
            if not self.fine_tune:
                EG_var_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'embedder')
            self.grads_EG = self.optimizer_EG.compute_gradients(self.loss_EG, var_list = EG_var_list)            
            ## gradient clipping
            self.clipped_EG = [(tf.clip_by_value(grad, -1., 1.) if not grad ==None else grad, var) for grad,var in self.grads_EG]
            self.train_EG = self.optimizer_EG.apply_gradients(self.clipped_EG) #, global_step=self.global_step)
            
            # Discriminator Optimization
            D_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')            
            self.grads_D = self.optimizer_D.compute_gradients(self.loss_DSC, var_list = D_var_list)            
            ## gradient clipping
            self.clipped_D = [(tf.clip_by_value(grad, -1., 1.)  if not grad ==None else grad,var) for grad,var in self.grads_D]
            # Updating Global step only during EG optimization as two optimization happens during trainig, so second incrementation of global step.
            self.train_D = self.optimizer_D.apply_gradients(self.clipped_D) #, global_step=self.global_step)

            # Global step increment
            self.global_step_increment =  tf.assign_add(self.global_step, 1, name = "global_step_increment")

            tf.summary.image('Generator/1/x', self.tx)
            tf.summary.image('Generator/2/y', self.ty)
            tf.summary.image('Generator/3/x_hat', self.x_hat)
            
            # Summary
            self.merged = tf.summary.merge_all()
        else:
            
            self.ty = tf.placeholder(tf.float32, [None] + list(self.img_size), name='ty')
            
            if not self.fine_tune:
                self.x = tf.placeholder(tf.float32, [None] + list(self.img_size), name='x')
                self.y = tf.placeholder(tf.float32, [None] + list(self.img_size), name='y')
                
                # Embedder
                # Calculate average encoding vector for video and AdaIn params input
                self.e_hat, self.psi_hat = self.Embedder(self.x, self.y, sn=True, reuse=False)
                
                # Generator
                # Generate frame using landmarks from frame t    
                self.x_hat = self.Generator(self.ty, psi_Pe=self.psi_hat, sn=True, reuse=False)
            else:
                # Generator
                # Generate frame using landmarks from frame t    
                self.x_hat = self.Generator(self.ty, psi_Pe=None, psi_hat_init = None, sn=True, reuse=False)
    
    def train(self):
        
        tf.logging.set_verbosity(tf.logging.INFO)
        
        self.load(self.logdir)
            
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        
        if not self.fine_tune:
            if os.path.exists(self.video_list_path):
                with open(self.video_list_path,'rb') as f:
                    video_list = pickle.load(f)
            else:
                video_list = get_video_list(self.dataset)
                with open(self.video_list_path,'rb') as f:
                    pickle.dump(video_list,f)
            while 1:                
                print("shuffling Dataset.....")                                
                idx = [i for i in range(0, len(video_list)-1)]
                random.shuffle(idx)                
                for v_id in idx:                    
                    vid_id = np.array(v_id).reshape(-1)
                    x, y, tx, ty = get_video_data(video_list[v_id])                    
                    if any([elem is None for elem in [x,y,tx,ty]]):
                        continue                    
                    feeddict = {self.x:x,self.y:y,self.tx:tx,self.ty:ty,self.idx:vid_id}                    
                    gs = self.training_operation(feeddict)
                    if gs > self.num_iterations: break
                if gs > self.num_iterations: break
        else:            
            while 1:                
                random.shuffle(self.frames)                
                for f in self.frames:                    
                    x, y, tx, ty = get_frame_data(f)                   
                    feeddict = {self.x:x,self.y:y,self.tx:tx,self.ty:ty}                    
                    gs = self.training_operation(feeddict)
                    if gs > self.num_iterations: break
                if gs > self.num_iterations: break
                    
        print("Training Completed..")
    
    def training_operation(self, feeddict):
        start_time = datetime.datetime.now()                    
        loss_dsc, _ = self.sess.run([self.loss_DSC, 
                                     self.train_D],
                                     feed_dict=feeddict)        
        loss_eg, _ = self.sess.run([self.loss_EG, 
                                    self.train_EG],
                                    feed_dict=feeddict)
        duration = datetime.datetime.now() - start_time
        
        summary, gs = self.sess.run([self.merged, self.global_step_increment],
                                    feed_dict=feeddict)
        
        if gs % self.log_step == 0:
            tf.logging.info('%s: step %d, loss_eg = %.5f, loss_dsc = %.5f (%.1f examples/sec; %.3f sec/batch)',
                            datetime.datetime.now(),gs,loss_eg,loss_dsc,(1/duration.total_seconds()),duration.total_seconds())                                                          
        # Write checkpoint files at every 1k steps
        if gs % self.save_step == 0:
            if not self.fine_tune:
                self.save(self.logdir,gs)
            else:
                self.save(self.fine_logdir,gs)                        
        if gs % self.summary_step == 0:
            self.writer.add_summary(summary, gs)

        return gs
    
    def save(self, logdir, gs):
        self.saver.save(self.sess, os.path.join(logdir, 'model.ckpt'), global_step=gs)
        tf.logging.info('Saving ckeckpoint at step %d',gs)
    
    def load(self, logdir):
        
        self.sess.run(tf.global_variables_initializer())
        
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        
        if not self.fine_tune:
            var_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'embedder')
        
        if self.training:
            var_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator') + \
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global_step')
        
        self.saver = tf.train.Saver(var_list)
        
        lastCheckpoint = tf.train.latest_checkpoint(logdir) 
        if lastCheckpoint is None:
            print("No checkpoint available.")
            pass
        else:
            print("Last checkpoint :", lastCheckpoint)
            self.saver.restore(self.sess, lastCheckpoint)
        
    
    def inference(self, ty, x = None, y = None):
                
        if not self.fine_tune:
            self.load(self.logdir)
            feeddict = {self.x:x,self.y:y,self.ty:ty}
        else:
            self.load(self.fine_logdir)
            feeddict = {self.ty:ty}
            
        frames = self.sess.run([self.x_hat], feed_dict=feeddict)
            
        return frames
    
    def update(self,newdata):
        for key,value in newdata.items():
            setattr(self,key,value)
