########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
import pandas as pd
from os import listdir
from os.path import isfile, join
from PIL import Image
from matplotlib import pyplot as plt

FEAT_DIR = './data/train/images'
LABELS_DIR = './data/train/masks'
WEIGHT_PATH = './vgg16/vgg16_weights.npz'
NUM_CLASSES = 1
EPOCHS = 10
BATCH_SIZE = 64
IMAGE_SHAPE = (101, 101, 3)
LABEL_SHAPE = (101, 101)
INPUT_SHAPE = (104, 104, 3)
OUTPUT_SHAPE = (104, 104)

class data_handling:
    
    def __init__(self, feat_path=FEAT_DIR, label_path=LABELS_DIR):
        self.create_arrays(feat_path, label_path)
        self.train_feat_data, \
        self.val_feat_data, \
        self.test_feat_data, \
        self.train_feat_data_filenames, \
        self.val_feat_data_filenames, \
        self.test_feat_data_filenames \
        = \
        self.split_data(self.feat_data, 
                        self.filenames)
        
        self.train_label_data, \
        self.val_label_data, \
        self.test_label_data, \
        _, _, _ = \
        self.split_data(self.label_data, 
                        self.filenames)
        
    def gen_batch_function(self, dataset='train',
                           bs=BATCH_SIZE, num_batches=None):
        
        if dataset == 'train':
            feat = self.train_feat_data
            labels = self.train_label_data
            
        elif dataset == 'test':
            feat = self.test_feat_data
            labels = self.test_label_data
            
        if num_batches is None:
            stop_iter = len(feat)//bs + 1
        else:
            stop_iter = num_batches
        
        batch = 0
        
        for i in range(stop_iter):
            if batch != len(feat)//bs:
                
                st = batch*bs; end = (batch+1)*bs;
                
                yield (feat[st:end,:].astype('float32')\
                - self.get_mean()), \
                labels[st:end,:].astype('float32')
                
                batch += 1
            else:
                yield feat[batch*bs:(len(feat)),:].astype('float32')\
                 - self.get_mean(), \
                labels[batch*bs:(len(feat)),:].astype('float32')

    def create_arrays(self, feat_path, label_path):
        
        files = [f for f in \
                       listdir(feat_path) if isfile(join(feat_path, f))]

        feat_data = np.zeros((len(files), *INPUT_SHAPE)).astype('int')
        label_data = np.zeros((len(files), *OUTPUT_SHAPE)).astype('int')
            
        for i in range(len(files)):
            feat_data[i,
                 :IMAGE_SHAPE[0],
                 :IMAGE_SHAPE[1],
                 :] = np.array(Image.open(feat_path + '/'+ files[i])) 
            label_data[i,
                 :IMAGE_SHAPE[0],
                 :IMAGE_SHAPE[1]
                ] = np.array(Image.open(label_path + '/'+ files[i]))                
        
        self.feat_data, self.label_data, self.filenames = self.shuffle(feat_data,
                                                 label_data,
                                                 pd.Series(files))
        
        self.label_data = self.label_data/65535.0
        
        return None
    
    @staticmethod
    def split_data(data, \
                   filenames, \
                   val_split = 0.10,
                   split = 0.8
                  ):
        
        train_end = int(len(data)*(split - val_split))
        val_end = int(len(data)*split)

        train_feat = data[:train_end]
        train_feat_filenames = filenames[:train_end]

        val_data = data[train_end:val_end]
        val_data_filenames = filenames[train_end:val_end]

        test_data = data[val_end:]
        test_data_filenames = filenames[val_end:]

        return train_feat, val_data, test_data, \
    train_feat_filenames, val_data_filenames, test_data_filenames
    
    
    @staticmethod
    def get_mean():
        x = np.zeros((1,1,1,3))
        x[0,0,0,:]= np.array([120.346, 120.346, 120.346])
        return x
    
    @staticmethod
    def get_std():
        return 27.60
    
    @staticmethod
    def shuffle(feat_data, label_data, filenames):
        ind = np.random.choice(len(feat_data),
                               len(feat_data),
                               replace=False
                              )
        return feat_data[ind], label_data[ind], filenames.loc[ind]
    