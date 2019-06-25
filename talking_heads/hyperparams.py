# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf

Hyperparams = tf.contrib.training.HParams(
        
        # Dataset
        dataset = "/media/new_hdd1/VoxCeleb-2/Video/dev/mp4",
               
        # No of training videos
        train_videos = 145569,
        
        # Network Architecture parameters
        # Encoder channels and self-attention channel
        enc_down_ch = [64,128,256,512],
        enc_self_att_ch = 256,
        
        # Decoder channels and self-attention channel
        dec_down_ch = [256,128,64,3],
        dec_self_att_ch = 64,
        
        # Residual Block channel
        res_blk_ch = 512,
        
        # Embedding Vector
        N_Vec = 512,
        
        # Considering input and output channel in a residual block, multiple of 2 because beta and gamma affine parameter.
        split_lens = [512]*11*2 + [256]*2*2 + [128]*2*2 + [64]*2*2 + [3]*2,
        
        # Activation outputs from VGGFace and VGG19
        vggface_feat_layers = ['conv1_1','conv2_1','conv2_1','conv3_1','conv5_1'],
        vgg19_feat_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'],
        
        # Training hyperparameters
        # Image Size
        img_size = (224, 224, 3),
        
        # K-shot learning,
        K = 8,
        
        # Loss weights
        loss_vgg19_wt =  1e-2,
        loss_vggface_wt = 2e-3,
        loss_fm_wt = 1e1,
        loss_mch_wt = 8e1,
        learning_rate_EG = 5e-5,
        learning_rate_D = 2e-4,
        epochs = 100
	
)

def hparams_debug_string():
    """
    Gives hyperparameters as string

    Returns:
        Hyperparameters
    """
    values = Hyperparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)