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

NUM_CLASSES = 1
WEIGHT_PATH = '/home/paperspace/kaggle/Semantic_Segmentation/vgg16/vgg16_weights.npz'
INPUT_SHAPE = (104, 104, 3)

class vgg16_modified:
    
    def __init__(self, imgs, dropout, phase, sess=None, weights=None):
        self.imgs = imgs
        self.phase = phase
        self.dropout = dropout
        self.output = self.layers()
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
        

    def layers(self, num_classes = NUM_CLASSES):
        self.parameters = []

        # zero-mean input if needed
        with tf.name_scope('preprocess') as scope:
            images = self.imgs
        
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(
                               self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1'
                                   )
        
        # dropout1
        self.pool1 = tf.layers.dropout(self.pool1, 
                                       rate=self.dropout,
                                       training=self.phase,
                                       name='dropout_1'
                                      )
        
        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')
        
        # dropout2
        self.pool2 = tf.layers.dropout(self.pool2, 
                                       rate=self.dropout,
                                       training=self.phase,
                                       name='dropout_2'
                                      ) 
        
        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # dropout3
        self.pool3 = tf.layers.dropout(self.pool3, 
                                       rate=self.dropout,
                                       training=self.phase,
                                       name='dropout_3'
                                      ) 
        
        # Use a shorter variable name for simplicity
        layer3, layer4, layer7 = self.pool1, self.pool2, self.pool3

        # Apply 1x1 convolution in place of fully connected layer
        self.fcn8 = tf.layers.conv2d(layer7,
                                     filters=num_classes,
                                     kernel_size=1,
                                     name="fcn8")


        # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
        self.fcn9 = tf.layers.conv2d_transpose(
            self.fcn8, filters=layer4.get_shape().as_list()[-1],
            kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9"
        )

        # Add a skip connection between current final layer fcn8 and 4th layer
        self.fcn9_skip_connected = tf.add(self.fcn9, layer4, name="fcn9_plus_vgg_layer4")

        # Upsample again
        self.fcn10 = tf.layers.conv2d_transpose(
            self.fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
            kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d"
        )

        # Add skip connection
        self.fcn10_skip_connected = tf.add(self.fcn10, layer3, name="fcn10_plus_vgg_layer3")

        # Upsample again
        self.fcn11 = tf.layers.conv2d_transpose(
            self.fcn10_skip_connected, filters=NUM_CLASSES,
            kernel_size=4, strides=(2, 2), padding='SAME', name="fcn11"
        )

        self.fcn11 = tf.identity(self.fcn11, name = 'final_output')
        
        return self.fcn11
                                    
    def load_weights(self, weight_file, sess):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print('Loading VGG16 Weights')
        print(' ')
        for i, k in enumerate(keys):
            if i < 14: #ensures only conv layers weights are uploaded
                print(i, k, np.shape(weights[k]))
                sess.run(self.parameters[i].assign(weights[k]))
        return None

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, *INPUT_SHAPE], name='image_input')
    imgs.get_shape()
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
    
