# -*- coding: utf-8 -*-
#/usr/bin/python3

import os
import re
from ast import literal_eval

class hyperparameters():

    def __init__(self):        
        # Dataset
        self.dataset = "/media/new_hdd1/VoxCeleb-2/Video/dev/mp4"
        self.data = "/media/new_hdd1/Face_Morp_2.0/Talking_Heads/data"
        self.preprocessed = os.path.join(self.data,"preprocessed")
        
        # logdir
        self.model = "fine"
        self.modeldir = "/media/new_hdd1/Face_Morp_2.0/Talking_Heads/results/model/"
        self.logdir = os.path.join(self.modeldir, "meta")
        self.fine_logdir = os.path.join(self.modeldir, self.model)
               
        # No of training videos
        self.train_videos = len(os.listdir(self.preprocessed))
        
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
        self.batch = 4
        
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
        
        # hyperparams json and resourceconfig json
        self.hp_json = "/opt/ml/input/config/hyperparameters.json"
        self.resource_json = "/opt/ml/input/config/resourceConfig.json"
        
    def update(self,newdata):
        for key,value in newdata.items():
            setattr(self,key,value)
            
Hyperparams = hyperparameters()

def hp_json(hp_json):
    '''Overrides hyperparams from hyperparameters.json'''
    print("READING ",Hyperparams.hp_json)
    with open(hp_json) as f:
        text = f.read()
        str_dict = re.sub(r"\"(-?\d+(?:[\.,]\d+)?)\"", r'\1', text)
        str_dict = str_dict.replace("\"True\"","True").replace("\"False\"","False")
        return literal_eval(str_dict)

def resource_json(resource_json):
    '''Overrides hyperparams from resourceConfig.json'''
    print("READING ",Hyperparams.resource_json)
    with open(resource_json) as f:
        text = f.read()
        str_dict = re.sub(r"\"(-?\d+(?:[\.,]\d+)?)\"", r'\1', text)
        str_dict = str_dict.replace("\"True\"","True").replace("\"False\"","False")
        return literal_eval(str_dict)

if os.path.exists(Hyperparams.hp_json):
    Hyperparams.update(hp_json(Hyperparams.hp_json))
else:
    Hyperparams.hp_json = 'hyperparameters.json'
    if os.path.exists(Hyperparams.hp_json):
        Hyperparams.update(hp_json(Hyperparams.hp_json))
        
if os.path.exists(Hyperparams.resource_json):
    Hyperparams.update(resource_json(Hyperparams.hp_json))
else:
    Hyperparams.resource_json = 'resourceConfig.json'
    if os.path.exists(Hyperparams.resource_json):
        Hyperparams.update(resource_json(Hyperparams.hp_json))

Hyperparams.logdir = os.path.join(Hyperparams.modeldir, "meta")        
Hyperparams.fine_logdir = os.path.join(Hyperparams.modeldir, Hyperparams.model)