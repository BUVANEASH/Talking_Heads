# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf
from keras_vggface.vggface import VGGFace
#from keras_vggface.utils import preprocess_tf_input
from hyperparams import Hyperparams as hp


def VGGFACE(input_tensor=None, input_shape = hp.img_size):

    vggface = VGGFace(input_tensor=input_tensor, model='vgg16', include_top=False, input_shape=input_shape)    
    outputs = []    
    for l in hp.vggface_feat_layers:
        outputs.append(vggface.get_layer(l).output)        
    model = tf.keras.Model(inputs = vggface.input, outputs = outputs)    
    
    return model

def VGG19(input_tensor=None, input_shape = hp.img_size):

    vgg19 = tf.keras.applications.VGG19(input_tensor=input_tensor, include_top=False, input_shape=input_shape)    
    outputs = []    
    for l in hp.vgg19_feat_layers:
        outputs.append(vgg19.get_layer(l).output)        
    model = tf.keras.Model(inputs = vgg19.input, outputs = outputs)
    
    return model
   
# =============================================================================
# Loss functions for Embedder, Generator and Discriminator training
# =============================================================================

def loss_cnt(xi, x_hat):
    
    vggface_xi = VGGFACE(input_tensor=xi, input_shape = hp.img_size) # preprocess_tf_input(tf.multiply(tf.add(xi,1.0),127.5))
    vggface_x_hat = VGGFACE(input_tensor=x_hat, input_shape = hp.img_size) # preprocess_tf_input(tf.multiply(tf.add(x_hat,1.0),127.5))
    
    vgg19_xi = VGG19(input_tensor=xi, input_shape = hp.img_size)
    vgg19_x_hat = VGG19(input_tensor=x_hat, input_shape = hp.img_size)
    
    vggface_loss = 0
    for i in range(len(vggface_xi.output)):
        vggface_loss  += tf.reduce_mean(tf.abs(vggface_xi.output[i] - vggface_x_hat.output[i]))
        
    vgg19_loss = 0
    for i in range(len(vgg19_xi.output)):
        vgg19_loss  += tf.reduce_mean(tf.abs(vgg19_xi.output[i] - vgg19_x_hat.output[i]))

    return vgg19_loss * hp.loss_vgg19_wt + vggface_loss * hp.loss_vggface_wt
      
def loss_fm(d_act, d_act_hat):
    loss = 0
    for i in range(0, len(d_act)):
        loss += tf.reduce_mean(tf.abs(d_act[i] - d_act_hat[i]))
    return loss * hp.loss_fm_wt
  
def loss_adv(r_x_hat, d_act, d_act_hat):
    return -r_x_hat + loss_fm(d_act, d_act_hat)
        
def loss_mch(e_hat, W_i):
    return tf.reduce_mean(tf.abs(W_i - e_hat)) * hp.loss_mch_wt

# =============================================================================
# Loss functions for Discriminator training specificailly
# =============================================================================
     
def loss_dsc(r_x, r_x_hat):
    return (1 + r_x_hat) + (1 - r_x)
   
