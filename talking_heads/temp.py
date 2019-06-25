#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:42:55 2019

@author: avantariml
"""
import tensorflow as tf
from network_ops import *
from utils import *
import cv2
import numpy as np
from KGAN import *
from hyperparams import Hyperparams as hp

x = tf.placeholder(tf.float32, shape=[None, 256,256,3], name = 'x')
y = tf.placeholder(tf.float32, shape=[None, 256,256,3], name = 'y')

sn=True
is_training=True
reuse=False

fine_tune=False
psi_hat_init = None
w_hat_init = None

train_videos = 143000

G = generator()
E = embedder()
D, W = discriminator()

e, psi_ = E(x,y)
x_hat = G(y, psi_Pe = psi_)
r, d_act = D(x,y,i=0)

channels = 256

tf.get_variable_scope().reuse_variables()

var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'discriminator')    

total_parameters = 0
for variable in var_list:
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    #print(shape)
    #print(len(shape))
    variable_parameters = 1
    for dim in shape:
        #print(dim)
        variable_parameters *= dim.value
    print(variable.name)
    total_parameters += variable_parameters
print(total_parameters)


cv2.imshow('tx',cv2.cvtColor(np.uint8((tx[0]+1)*127.50), cv2.COLOR_BGR2RGB))
cv2.imshow('ty',cv2.cvtColor(np.uint8((ty[0]+1)*127.50), cv2.COLOR_BGR2RGB))
for i in range(x.shape[0]):
    cv2.imshow('x{}'.format(i),cv2.cvtColor(np.uint8((x[i]+1)*127.50), cv2.COLOR_BGR2RGB))
    cv2.imshow('y{}'.format(i),cv2.cvtColor(np.uint8((y[i]+1)*127.50), cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()

d = data()

with tf.Session() as sess:
    output = sess.run(d)

W = tf.get_variable("W1", shape=[hp.train_videos,hp.N_Vec], initializer=weight_init, regularizer=weight_regularizer)
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())
    dum = tf.reshape(tf.nn.embedding_lookup(W,1), shape=[-1,1]).eval()
    
# =============================================================================
# 
# =============================================================================


video_list = get_video_list(hp.dataset)

idx = random.randint(0,len(video_list))

x, y, tx, ty = get_video_data(video_list[idx])                

idx =np.int32(idx)
x = np.float32(x)
y = np.float32(y)
tx = np.float32(tx)
ty = np.float32(ty)

g = Graph()

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
sess = tf.Session()
k = KGAN(sess=sess)

if k.training:
    # Training Scheme
    k.learning_rate_EG = learning_rate_decay(k.learning_rate_EG, k.global_step)
    k.optimizer_EG = tf.train.AdamOptimizer(learning_rate=k.learning_rate_EG)
    tf.summary.scalar("learning_rate_EG", k.learning_rate_EG)
    
   
    k.learning_rate_D = learning_rate_decay(k.learning_rate_D, k.global_step)
    k.optimizer_D = tf.train.AdamOptimizer(learning_rate=k.learning_rate_D)
    tf.summary.scalar("learning_rate_D", k.learning_rate_D)
    
    
        
    k.x = tf.placeholder(tf.float32, [None] + list(k.img_size), name='x')
    k.y = tf.placeholder(tf.float32, [None] + list(k.img_size), name='y')
    k.tx = tf.placeholder(tf.float32, [None] + list(k.img_size), name='tx')
    k.ty = tf.placeholder(tf.float32, [None] + list(k.img_size), name='ty')
    if not k.fine_tune:
        k.idx = tf.placeholder(tf.int32, [1,], name='idx')
        
    # Embedder
    # Calculate average encoding vector for video and AdaIn params input
    k.e_hat, k.psi_hat = k.Embedder(k.x, k.y, sn=True, reuse=False)
    
    if not k.fine_tune:
        # Generator
        # Generate frame using landmarks from frame t    
        k.x_hat = k.Generator(k.ty, psi_Pe=k.psi_hat, sn=True, reuse=False)           
        # Discriminator
        # real score for fake image
        k.r_x_hat, k.D_act_hat = k.Discriminator(k.x_hat, k.ty, i=k.idx, e_new = None, sn=True, reuse=False)
        # real score for real image
        k.r_x, k.D_act =k.Discriminator(k.x, k.y, i=k.idx, e_new = None, sn=True, reuse=True)
    else:
                              
        x, y, _, _ = get_frame_data(k.frames)
        
        embedder_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'embedder')
        
        embedder_saver = tf.train.Saver(var_list=embedder_var_list)
        embedder_saver.restore(k.sess, tf.train.latest_checkpoint(k.logdir))
        
        k.sess.run(tf.global_variables_initializer)
        
        e_hat, psi_hat = k.sess.run([k.e_hat, k.psi_hat],
                                       feeddict = {k.x:x,
                                                   k.y:y})
        
        # Generator
        # Generate frame using landmarks from frame t    
        k.x_hat = k.Generator(k.ty, psi_Pe=None, psi_hat_init = psi_hat, sn=True, reuse=False)           
        # Discriminator
        # real score for fake image
        k.r_x_hat, k.D_act_hat = k.Discriminator(k.x_hat, k.ty, i=None, e_new = e_hat, sn=True, reuse=False)
        # real score for real image
        k.r_x, k.D_act =k.Discriminator(k.x, k.y, i=None, e_new = e_hat, sn=True, reuse=True)
    
    k.loss_CNT = k.loss_cnt(k.tx, k.x_hat)
    k.loss_ADV = k.loss_adv(k.r_x_hat, k.D_act, k.D_act_hat)
    k.loss_DSC = k.loss_dsc(k.r_x,k.r_x_hat)
        
    if not k.fine_tune:
        k.loss_MCH = k.loss_mch(k.e_hat, tf.reshape(tf.nn.embedding_lookup(k.W,k.idx), shape=[-1,1]))
        
        k.loss_EG =  k.loss_CNT + k.loss_ADV + k.loss_MCH 
    else:
        k.loss_EG =  k.loss_CNT + k.loss_ADV
        
    tf.summary.scalar("loss_CNT", k.loss_CNT)
    tf.summary.scalar("loss_ADV", k.loss_ADV)
    if not k.fine_tune:
        tf.summary.scalar("loss_MCH", k.loss_MCH)
    tf.summary.scalar("loss_EG", k.loss_EG)
    tf.summary.scalar("loss_DSC", k.loss_DSC)
    
    
    
    EG_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'embedder') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    
        
    k.grads_EG = k.optimizer_EG.compute_gradients(k.loss_EG, var_list = EG_var_list)            
    ## gradient clipping
    k.clipped_EG = [(tf.clip_by_value(grad, -1., 1.) if not grad ==None else grad, var) for grad,var in k.grads_EG]            
    k.train_EG = k.optimizer_EG.apply_gradients(k.clipped_EG, global_step=k.global_step)
    
    D_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    
    k.grads_D = k.optimizer_D.compute_gradients(k.loss_DSC, var_list = D_var_list)  
    ## gradient clipping
    k.clipped_D = [(tf.clip_by_value(grad, -1., 1.)  if not grad ==None else grad,var) for grad,var in k.grads_D]
    k.train_D = k.optimizer_D.apply_gradients(k.clipped_D, global_step=k.global_step)
    
    tf.summary.image('Generator/in/y', k.ty)
    tf.summary.image('Generator/out/x_hat', k.x_hat)
    tf.summary.image('Embedder/y', k.y[np.random.randint(0,k.K+1)])
    tf.summary.image('Embedder/x', k.x[np.random.randint(0,k.K+1)])
    
    # Summary
    k.merged = tf.summary.merge_all()

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

config = tf.ConfigProto()
config.allow_soft_placement = True
config.log_device_placement = False
config.gpu_options.allow_growth = True
    
sess = tf.Session(config=config)
k = KGAN(sess=sess)

k.build()

tf.logging.set_verbosity(tf.logging.INFO)
k.sess.run(tf.global_variables_initializer())

if not k.fine_tune:
    
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'embedder') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator') + \
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global_step')
    saver = tf.train.Saver(var_list)
    writer = tf.summary.FileWriter(k.logdir, k.sess.graph)
    lastCheckpoint = tf.train.latest_checkpoint(k.logdir) 
    if lastCheckpoint is None:
        pass
    else:
        print("Last checkpoint :", lastCheckpoint)
        saver.restore(k.sess, lastCheckpoint)
    
    video_list = get_video_list(k.dataset)
    
    for ep in k.epoch:
        
        print("Epoch {0}/{1}".format(ep,k.epoch))
        
        
        idx = [i for i in range(0, len(video_list)-1)]
        random.shuffle(idx)
        
        for v_id in idx:
            #v_id = random.choice(idx)
            vid_id = np.array(v_id).reshape(-1)
            x, y, tx, ty = get_video_data(video_list[v_id])
            if None in [x,y,tx,ty]:
                continue
            
            start_time = datetime.datetime.now()
            
            loss_dsc, _ = k.sess.run([k.loss_DSC, 
                                         k.train_D],
                                         feed_dict={k.x:x,
                                                   k.y:y,
                                                   k.tx:tx,
                                                   k.ty:ty,
                                                   k.idx:vid_id})

            
            loss_eg, _ = k.sess.run([k.loss_EG, 
                                        k.train_EG],
                                        feed_dict={k.x:x,
                                                  k.y:y,
                                                  k.tx:tx,
                                                  k.ty:ty,
                                                   k.idx:vid_id})

            duration = datetime.datetime.now() - start_time
            
            summary, gs = k.sess.run([k.merged, k.global_step],
                                        feed_dict={k.x:x,
                                                  k.y:y,
                                                  k.tx:tx,
                                                  k.ty:ty,
                                                   k.idx:vid_id})
            if gs % 10 == 0:
                examples_per_sec = 1 / duration.total_seconds()
                sec_per_batch = duration.total_seconds()
                tf.logging.info('%s: step %d, loss_eg = %.5f, loss_dsc = %.5f (%.1f examples/sec; %.3f sec/batch)',
                                datetime.datetime.now(),gs,loss_eg,loss_dsc,examples_per_sec,sec_per_batch)                                                          
            # Write checkpoint files at every 1k steps
            if gs % 1000 == 0:
                saver.save(k.sess, os.path.join(k.logdir, 'model.ckpt'), global_step=gs)
                tf.logging.info('Saving ckeckpoint at step %d',gs)
            if gs % 100 == 0:
                writer.add_summary(summary, gs)
# =============================================================================
# 
# =============================================================================

video_list = get_video_list(hp.dataset)

idx = [i for i in range(0, len(video_list)-1)]
random.shuffle(idx)
        
v_id = random.choice(idx)
vid_id = np.array(v_id).reshape(-1)
x, y, tx, ty = get_video_data(video_list[v_id])
            
sn = True
z = tf.placeholder(tf.float32, [13062], name='z')
yy = tf.placeholder(tf.float32, [None,224,224,3], name='yy')

z_splits = tf.split(z, num_or_size_splits=hp.split_lens)
                
x_rd1 = resblock_down(yy, channels=hp.enc_down_ch[0], use_bias=True, is_training=True, sn=sn, scope='resblock_down_1')
x_rd2 = resblock_down(x_rd1, channels=hp.enc_down_ch[1], use_bias=True, is_training=True, sn=sn, scope='resblock_down_2')
x_rd3 = resblock_down(x_rd2, channels=hp.enc_down_ch[2], use_bias=True, is_training=True, sn=sn, scope='resblock_down_3')
x_s1 = self_attention_2(x_rd3, channels=hp.enc_self_att_ch, sn=sn, scope='self_attention_down')
x_rd4 = resblock_down(x_s1, channels=hp.enc_down_ch[3], use_bias=True, is_training=True, sn=sn, scope='resblock_down_4')

x_r1 = resblock_condition(x_rd4, z_splits[:4], channels=hp.res_blk_ch, use_bias=True, is_training=True, sn=sn, scope='resblock_1')
x_r2 = resblock_condition(x_r1, z_splits[4:8], channels=hp.res_blk_ch, use_bias=True, is_training=True, sn=sn, scope='resblock_2')
x_r3 = resblock_condition(x_r2, z_splits[8:12], channels=hp.res_blk_ch, use_bias=True, is_training=True, sn=sn, scope='resblock_3')
x_r4 = resblock_condition(x_r3, z_splits[12:16], channels=hp.res_blk_ch, use_bias=True, is_training=True, sn=sn, scope='resblock_4')
x_r5 = resblock_condition(x_r4, z_splits[16:20], channels=hp.res_blk_ch, use_bias=True, is_training=True, sn=sn, scope='resblock_5')

x_ru1 = resblock_up_condition(x_r5, z_splits[20:24], channels=hp.dec_down_ch[0], use_bias=True, is_training=True, sn=sn, scope='resblock_up_1')
x_ru2 = resblock_up_condition(x_ru1, z_splits[24:28], channels=hp.dec_down_ch[1], use_bias=True, is_training=True, sn=sn, scope='resblock_up_2')
x_ru3 = resblock_up_condition(x_ru2, z_splits[28:32], channels=hp.dec_down_ch[2], use_bias=True, is_training=True, sn=sn, scope='resblock_up_3')
x_s2 = self_attention_2(x_ru3, channels=hp.dec_self_att_ch, sn=sn, scope='self_attention_up')
x_ru4 = resblock_up_condition(x_s2, z_splits[32:36], channels=hp.dec_down_ch[3], use_bias=True, is_training=True, sn=sn, scope='resblock_up_4')

x = tanh(x_ru4)


with tf.variable_scope('res1'):
    x_ru1_a1 = adaptive_instance_norm(x_r5, z_splits[20:22], True)
    x_ru1_r1 = relu(x_ru1_a1)
    x_ru1_d1 = deconv(x_ru1_r1, hp.dec_down_ch[0], kernel=3, stride=2, use_bias=True, sn=sn)

with tf.variable_scope('res2') :
    x_ru1_a2 = adaptive_instance_norm(x_ru1_d1, z_splits[22:24], True)
    x_ru1_r2 = relu(x_ru1_a2)
    x_ru1_d2 = deconv(x_ru1_r2, hp.dec_down_ch[0], kernel=3, stride=1, use_bias=True, sn=sn)

with tf.variable_scope('skip') :
    x_s = deconv(x_r5, hp.dec_down_ch[0], kernel=3, stride=2, use_bias=True, sn=sn)
    
ru_out = x_ru1_d2 + x_s

sn =True
use_bias=True 
is_training=True
stride = 2
kernel=3
padding = 'SAME'
channels = hp.dec_down_ch[0]

dx = x_ru1_r1

with tf.variable_scope('deconv_0'):
    dx_shape = dx.get_shape().as_list()

    if padding == 'SAME':
        output_shape = [-1, dx_shape[1] * stride, dx_shape[2] * stride, channels]

    else:
        output_shape =[-1, dx_shape[1] * stride + max(kernel - stride, 0), dx_shape[2] * stride + max(kernel - stride, 0), channels]

    
    w = tf.get_variable("kernel", shape=[kernel, kernel, channels, dx_shape[-1]], initializer=weight_init, regularizer=weight_regularizer)
    
    if sn:
        dtx = tf.nn.conv2d_transpose(dx, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)
    else:
        dtx = tf.nn.conv2d_transpose(dx, filter=w, output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

    if use_bias :
        bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
        dtxb = tf.nn.bias_add(dtx, bias)

output_shape = [9, dx_shape[1] * stride, dx_shape[2] * stride, channels]

sess = tf.Session()

sess.run(tf.global_variables_initializer())

out = sess.run([dtx], feed_dict = {yy:y,z:np.random.rand(13062)})[0]


padding = 'SAME'

dxt = tf.placeholder(tf.float32, [None, 14, 14, 512], name='dxt')

with tf.variable_scope('deconv_0'):
    dxt_shape = dxt.get_shape().as_list()
    
    if padding == 'SAME':
        output_shape = tf.stack([tf.shape(dxt)[0], dxt_shape[1] * stride, dxt_shape[2] * stride, channels])
    
    else:
        output_shape = tf.stack([tf.shape(dxt)[0], dxt_shape[1] * stride + max(kernel - stride, 0), dxt_shape[2] * stride + max(kernel - stride, 0), channels])
    
    
    wt = tf.get_variable("kernel", shape=[kernel, kernel, channels, dxt_shape[-1]], initializer=weight_init, regularizer=weight_regularizer)
    
    if sn:
        dtxt = tf.nn.conv2d_transpose(dxt, filter=spectral_norm(wt), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)
    else:
        dtxt = tf.nn.conv2d_transpose(dxt, filter=wt, output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)
    
    if use_bias :
        bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
        dtxtb = tf.nn.bias_add(dtxt, bias)
        
sess = tf.Session()

sess.run(tf.global_variables_initializer())

out = sess.run([dtxt], feed_dict = {dxt:np.random.rand(1, 14, 14, 512)})[0]
# 9,14,14,512

output = tf.constant(0.1, shape = [1, 28, 28, 256])

exp_l = tf.nn.conv2d(output, 
                     tf.constant(0.1, shape = [3,3,256,256]),
                     strides = [1,2,2,1],
                     padding = 'SAME')



output_shape = [-1, (dxt_shape[1]-1) * stride +kernel, (dxt_shape[2]-1) * stride+kernel, channels]



