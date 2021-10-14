import tensorflow as tf
import numpy as np
from cnn import *

class CAAE:
    def __init__(self, n_labels, z_dim):
        self.n_labels = n_labels
        self.z_dim = z_dim
    

    def dense(self, x, n1, n2, name):
        """
        Used to create a dense layer.
        :param x: input tensor to the dense layer
        :param n1: no. of input neurons
        :param n2: no. of output neurons
        :param name: name of the entire dense layer.i.e, variable scope name.
        :return: tensor with shape [batch_size, n2]
        """
        with tf.variable_scope(name, reuse=None):
            weights = tf.get_variable("weights", shape=[n1, n2],
                                      initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
            bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
            out = tf.add(tf.matmul(x, weights), bias, name='matmul')
            return out
        
    def encoder(self, x, keep_prob, reuse=False, supervised=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Encoder'):
            x_input = tf.reshape(x, shape=[-1, 29, 29, 1]) #[batch_size, 29*29] -> [batch_size, 29, 29, 1]
            x_input = tf.pad(x_input, [[0, 0], [1, 2], [1, 2], [0, 0]]) #[batch_size, 29, 29, 1] -> [batch_size, 32, 32, 1]
        
            #print('Input encoder: ', x_input.shape)
            conv1 = conv2d(x_input, name='e_conv1', kshape=[3, 3, 1, 32]) #[32, 32, 1] -> [32, 32, 32]
            pool1 = maxpool2d(conv1, name='e_pool1') # [32, 32 32] -> [16, 16, 32]

            conv2 = conv2d(pool1, name='e_conv2', kshape=[3, 3, 32, 32]) #[16, 16, 32] -> [16, 16, 32]
            pool2 = maxpool2d(conv2, name='e_pool2') # [16, 16, 32] -> [8, 8, 32]

            conv3 = conv2d(pool2, name='e_conv3', kshape=[3, 3, 32, 64]) #[8, 8, 32] -> [8, 8, 64]
            pool3 = maxpool2d(conv3, name='e_pool3') # [8, 8, 64] -> [4, 4, 64]

            conv4 = conv2d(pool3, name='e_conv4', kshape=[3, 3, 64, 64]) #[4, 4, 64] -> [4, 4, 64]
            pool4 = maxpool2d(conv4, name='e_pool4') # [4, 4, 64] -> [2, 2, 64]
            drop4 = dropout(pool4, name='e_drop4', keep_rate=keep_prob)

            latent_variable = fullyConnected(drop4, name='e_latent_variable', output_size=self.z_dim)
            cat_op = fullyConnected(drop4, name='e_label', output_size=self.n_labels)

            if not supervised:
                softmax_label = tf.nn.softmax(logits=cat_op, name='e_softmax_label')
            else:
                softmax_label = cat_op
            return softmax_label, latent_variable
    
    def decoder(self, x, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Decoder'):
            fc1 = fullyConnected(x, name='d_fc1', output_size=2*2*64)
            fc1 = tf.reshape(fc1, shape=[-1, 2, 2, 64])

            deconv1 = deconv2d(fc1, name='d_deconv1', kshape=[3, 3], n_outputs=64) #[2, 2, 64] -> [2, 2, 64]
            up1 = upsample(deconv1, name='d_up1', factor=[2, 2]) #[2, 2, 64] -> [4, 4, 64]

            deconv2 = deconv2d(up1, name='d_deconv2', kshape=[3, 3], n_outputs=32) #[4, 4, 64] -> [4, 4, 32]
            up2 = upsample(deconv2, name='d_up2', factor=[2, 2]) #[4, 4, 32] -> [8, 8, 32]

            deconv3 = deconv2d(up2, name='d_deconv3', kshape=[3, 3], n_outputs=32) #[8, 8, 32] -> [8, 8, 32] 
            up3 = upsample(deconv3, name='d_up3', factor=[2, 2]) #[8, 8, 32] -> [16, 16, 32] 

            deconv4 = deconv2d(up3, name='d_deconv4', kshape=[3, 3], n_outputs=1) #[16, 16, 32] -> [16, 16, 1]
            up4 = upsample(deconv4, name='d_up4', factor=[2, 2]) #[16, 16, 1] -> [32, 32, 1]

            out = tf.image.crop_to_bounding_box(up4, 1, 1, 29, 29)
            out = tf.reshape(out, shape=[-1, 29 * 29])
            
            return out

    def discriminator_gauss(self, x, reuse=False):
        """
        Discriminator that is used to match the posterior distribution with a given gaussian distribution.
        :param x: tensor of shape [batch_size, z_dim]
        :param reuse: True -> Reuse the discriminator variables,
                      False -> Create or search of variables before creating
        :return: tensor of shape [batch_size, 1]
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Discriminator_Gauss'):
            dc_den1 = tf.nn.relu(self.dense(x, self.z_dim, 1000, name='dc_g_den1'))
            dc_den2 = tf.nn.relu(self.dense(dc_den1, 1000, 1000, name='dc_g_den2'))
            output = self.dense(dc_den2, 1000, 1, name='dc_g_output')
            return tf.nn.sigmoid(output)

    def discriminator_categorical(self, x, reuse=False):
        """
        Discriminator that is used to match the posterior distribution with a given categorical distribution.
        :param x: tensor of shape [batch_size, n_labels]
        :param reuse: True -> Reuse the discriminator variables,
                      False -> Create or search of variables before creating
        :return: tensor of shape [batch_size, 1]
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Discriminator_Categorial'):
            dc_den1 = tf.nn.relu(self.dense(x, self.n_labels, 1000, name='dc_c_den1'))
            dc_den2 = tf.nn.relu(self.dense(dc_den1, 1000, 1000, name='dc_c_den2'))
            output = self.dense(dc_den2, 1000, 1, name='dc_c_output')
            return tf.nn.sigmoid(output)