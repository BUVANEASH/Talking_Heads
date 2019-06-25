# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf
from network_ops import weight_init, weight_regularizer
from network_ops import resblock_down, resblock, resblock_condition, resblock_up_condition, self_attention_2
from network_ops import fully_connected, global_sum_pooling, relu, tanh
from hyperparams import Hyperparams as hp

##################################################################################
# Embedder
##################################################################################

def embedder(sn=True, is_training=True, reuse=False):
	    
    def Embedder_Net(x,y):
        with tf.variable_scope("embedder", reuse=reuse):
            with tf.variable_scope("embedder_ConvNet", reuse=reuse):
                e = tf.concat([x,y], axis=-1)
                e = resblock_down(e, channels=hp.enc_down_ch[0], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_1')
                e = resblock_down(e, channels=hp.enc_down_ch[1], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_2')
                e = resblock_down(e, channels=hp.enc_down_ch[2], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_3')
                e = self_attention_2(e, channels=hp.enc_self_att_ch, sn=sn, scope='self_attention')
                e = resblock_down(e, channels=hp.enc_down_ch[3], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_4')
                #e = resblock_down(e, channels=hp.enc_down_ch[3], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_5')
                
                e = global_sum_pooling(e)
                e = relu(e)
                e =  tf.reduce_sum(e, axis=0, keepdims=True)
                
            psi_hat = fully_connected(tf.expand_dims(tf.reduce_mean(e,axis=0),axis=0), units=sum(hp.split_lens) , use_bias=True, is_training=is_training, sn=sn, scope='P')
            
            return e, psi_hat
    
    return Embedder_Net

##################################################################################
# Generator
##################################################################################

def generator(sn=True, is_training=True, reuse=False, fine_tune=False, psi_hat_init = None):        
            
    def img2img(y, z):
        with tf.variable_scope("image-to-image", reuse=reuse):
            z_splits = tf.split(z, num_or_size_splits=hp.split_lens)
            
            x = resblock_down(y, channels=hp.enc_down_ch[0], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_1')
            x = resblock_down(x, channels=hp.enc_down_ch[1], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_2')
            x = resblock_down(x, channels=hp.enc_down_ch[2], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_3')
            x = self_attention_2(x, channels=hp.enc_self_att_ch, sn=sn, scope='self_attention_down')
            x = resblock_down(x, channels=hp.enc_down_ch[3], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_4')
            
            x = resblock_condition(x, z_splits[:4], channels=hp.res_blk_ch, use_bias=True, is_training=is_training, sn=sn, scope='resblock_1')
            x = resblock_condition(x, z_splits[4:8], channels=hp.res_blk_ch, use_bias=True, is_training=is_training, sn=sn, scope='resblock_2')
            x = resblock_condition(x, z_splits[8:12], channels=hp.res_blk_ch, use_bias=True, is_training=is_training, sn=sn, scope='resblock_3')
            x = resblock_condition(x, z_splits[12:16], channels=hp.res_blk_ch, use_bias=True, is_training=is_training, sn=sn, scope='resblock_4')
            x = resblock_condition(x, z_splits[16:20], channels=hp.res_blk_ch, use_bias=True, is_training=is_training, sn=sn, scope='resblock_5')
            
            x = resblock_up_condition(x, z_splits[20:24], channels=hp.dec_down_ch[0], use_bias=True, is_training=is_training, sn=sn, scope='resblock_up_1')
            x = resblock_up_condition(x, z_splits[24:28], channels=hp.dec_down_ch[1], use_bias=True, is_training=is_training, sn=sn, scope='resblock_up_2')
            x = resblock_up_condition(x, z_splits[28:32], channels=hp.dec_down_ch[2], use_bias=True, is_training=is_training, sn=sn, scope='resblock_up_3')
            x = self_attention_2(x, channels=hp.dec_self_att_ch, sn=sn, scope='self_attention_up')
            x = resblock_up_condition(x, z_splits[32:36], channels=hp.dec_down_ch[3], use_bias=True, is_training=is_training, sn=sn, scope='resblock_up_4')
            
            x = tanh(x)
            
            return x
        
    def gen_net(y, psi_Pe=None):
        with tf.variable_scope("generator", reuse=reuse):
            
            psi_hat = tf.get_variable("AdaIN_params", shape=[sum(hp.split_lens)], initializer=psi_hat_init if (is_training and fine_tune) else weight_init, 
                                    regularizer=weight_regularizer, 
                                    trainable=True if (is_training and fine_tune) else False)

            if not fine_tune:
                Pe = tf.assign(psi_hat, tf.squeeze(psi_Pe))
                with tf.control_dependencies([Pe]):
                    x = img2img(y,psi_hat)
                    return x
            else:
                x = img2img(y,psi_hat)
                return x
                
    return gen_net

##################################################################################
# Discriminator
##################################################################################

def discriminator(sn=True, is_training=True, reuse=False, fine_tune=False, e_new = None):
    
    with tf.variable_scope("discriminator_params", reuse=reuse):
        W = tf.get_variable("W", shape=[hp.train_videos,hp.N_Vec], initializer=weight_init, regularizer=weight_regularizer)
        w_0 = tf.get_variable("w_0", shape=[hp.N_Vec,1], initializer=weight_init, regularizer=weight_regularizer)
    
        def Discriminator_Net(x,y):
            with tf.variable_scope("discriminator_ConvNet", reuse=reuse):
                c = tf.concat([x,y], axis=-1)
                o0 = resblock_down(c, channels=hp.enc_down_ch[0], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_1')
                o1 = resblock_down(o0, channels=hp.enc_down_ch[1], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_2')
                o2 = resblock_down(o1, channels=hp.enc_down_ch[2], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_3')
                o3 = self_attention_2(o2, channels=hp.enc_self_att_ch, sn=sn, scope='self_attention')
                o4 = resblock_down(o3, channels=hp.enc_down_ch[3], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_4')
                #o5 = resblock_down(o4, channels=hp.enc_down_ch[3], use_bias=True, is_training=is_training, sn=sn, scope='resblock_down_5')
                            
                o = resblock(o4, channels=hp.N_Vec, kernel=4, pads=[2,1], use_bias=True, is_training=is_training, sn=sn, scope='resblock')
                
                o = global_sum_pooling(o)
                o = relu(o)
                return o, [o0,o1,o2,o3,o4] #,o5]
        
        def discriminator_block(x, y, i=None):            
            with tf.variable_scope("discriminator", reuse=reuse):
                
                w_hat = tf.get_variable("w_hat", shape=[hp.N_Vec,1], initializer=tf.math.add(tf.reshape(e_new,shape=[-1,1]), w_0) if (is_training and fine_tune) else weight_init, 
                                        regularizer=weight_regularizer, 
                                        trainable=True if (is_training and fine_tune) else False)
        
                b = tf.get_variable("bias", shape=[1], initializer=weight_init, regularizer=weight_regularizer)
                
                o , d_act = Discriminator_Net(x,y)
                
                if not fine_tune:
                    w_assign = tf.assign(w_hat, tf.math.add(tf.reshape(tf.nn.embedding_lookup(W,i), shape=[-1,1]), w_0))
                    with tf.control_dependencies([w_assign]):
                        r = tanh(tf.math.add(tf.matmul(o, w_hat), b))
                        return r, d_act
                else:
                    r = tanh(tf.math.add(tf.matmul(o, w_hat), b))
                    return r, d_act
    
    if not fine_tune:
        return discriminator_block, W
    else:
        return discriminator_block
