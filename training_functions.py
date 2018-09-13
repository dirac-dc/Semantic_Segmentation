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
from tensorflow.python import debug as tf_debug
import pandas as pd
from os import listdir
from os.path import isfile, join
from PIL import Image

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

def optimize(nn_last_layer, correct_label, learning_rate = LRATE, num_classes = NUM_CLASSES):
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss") # actual loss value
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")
    return logits, train_op, loss_op

def train_nn(sess, epochs, batch_size, data_handler, train_op,
             cross_entropy_loss, input_image,
             correct_label, phase_ph):
    
    def check_and_delete_existing(directory):
        if os.path.exists(directory):
            os.system("rm -rf "+directory)
        return directory
    
    output_path = check_and_delete_existing("./Train")
    train_summary_writer = tf.summary.FileWriter(output_path)
    
    train_summary=tf.Summary()
    val_summary=tf.Summary()
    
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    total_train_loss = 0
    total_val_loss = 0
    step = 0 
    for epoch in range(epochs):
        total_train_loss = 0
        total_val__loss = 0
        for X_batch, gt_batch in data_handler.gen_batch_function(bs = batch_size):
            step += 1
            
            loss, _ = sess.run([cross_entropy_loss, train_op], 
                               feed_dict={input_image: X_batch, 
                                          correct_label: gt_batch,
                                          phase_ph: 1})
            
            val_loss = sess.run([cross_entropy_loss], 
                                feed_dict={input_image: data_handler.val_feat_data, 
                                           correct_label: data_handler.val_label_data, 
                                           phase_ph: 1})
            
            train_summary.value.add(tag='train_loss', simple_value = loss)
            val_summary.value.add(tag='val_loss', simple_value = val_loss[0])
            train_summary_writer.add_summary(train_summary, step)
            train_summary_writer.add_summary(val_summary, step)
            
            # train_summary_writer.flush()
            total_train_loss += loss;
            total_val_loss += val_loss[0]
        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f};  Val_loss = {:.3f}".format(total_train_loss, total_val_loss))
        print()
    
    graph = tf.get_default_graph()
    
    output = graph.get_tensor_by_name('final_output:0')
    
    train_pred = sess.run([output], 
             feed_dict={input_image: data_handler.train_feat_data[:5], 
                        correct_label: data_handler.train_label_data[:5], 
                        phase_ph: 0})
    
    test_pred = sess.run([output], 
             feed_dict={input_image: data_handler.val_feat_data[:5], 
                        correct_label: data_handler.val_label_data[:5], 
                        phase_ph: 0})
    
    return (data_handler.train_feat_data[:5], 
            train_pred, data_handler.train_label_data[:5], 
            data_handler.val_feat_data[:5],
            test_pred,
            data_handler.val_label_data[:5]
           )