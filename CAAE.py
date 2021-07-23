import tensorflow as tf
import numpy as np
from cnn import *

class CAAE:
    def __init__(self, n_labels, z_dim):
        self.n_labels = n_labels
        self.z_dim = z_dim
    

    def encoder(self, x, reuse=False, supervised=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Encoder'):
            x_input = tf.reshape(x, shape=[-1, 29, 29, 1]) #[batch_size, 29*29] -> [batch_size, 29, 29, 1]
            x_input = tf.pad(x_input, [[0, 0], [1, 2], [1, 2], [0, 0]]) #[batch_size, 29, 29, 1] -> [batch_size, 32, 32, 1]
        
            #print('Input encoder: ', x_input.shape)
            conv1 = conv2d(x_input, name='e_conv1', kshape=[3, 3, 1, 32]) #[32, 32, 1] -> [32, 32, 32]
            pool1 = maxpool2d(conv1, name='e_pool1') # [32, 32 32] -> [16, 16, 32]
            drop1 = dropout(pool1, name='e_drop1', keep_rate=0.75)

            conv2 = conv2d(drop1, name='e_conv2', kshape=[3, 3, 32, 32]) #[16, 16, 32] -> [16, 16, 32]
            pool2 = maxpool2d(conv2, name='e_pool2') # [16, 16, 32] -> [8, 8, 32]
            drop2 = dropout(pool2, name='e_drop2', keep_rate=0.75)

            conv3 = conv2d(drop2, name='e_conv3', kshape=[3, 3, 32, 64]) #[8, 8, 32] -> [8, 8, 64]
            pool3 = maxpool2d(conv3, name='e_pool3') # [8, 8, 64] -> [4, 4, 64]
            drop3 = dropout(pool3, name='e_drop3', keep_rate=0.75)

            conv4 = conv2d(drop3, name='e_conv4', kshape=[3, 3, 64, 64]) #[4, 4, 64] -> [4, 4, 64]
            pool4 = maxpool2d(conv4, name='e_pool4') # [4, 4, 64] -> [2, 2, 64]
            #pool4 = tf.reshape(pool4, shape=[-1, 2 * 2 * 64])

            #print('Last CNN encoder: ', pool4.shape)
            latent_variable = fullyConnected(pool4, name='e_latent_variable', output_size=self.z_dim)
            cat_op = fullyConnected(pool4, name='e_label', output_size=self.n_labels)

            if not supervised:
                softmax_label = tf.nn.softmax(logits=cat_op, name='e_softmax_label')
            else:
                softmax_label = cat_op
            return softmax_label, latent_variable
    
    def decoder(self, x, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Decoder'):
            #print('Input decoder: ', x.shape)
            #encoded = tf.reshape(x, shape=[-1, 1, z_dim + n_labels])
        
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

            #print('Output decoder: ', up4.shape)
            out = tf.image.crop_to_bounding_box(up4, 1, 1, 29, 29)
            out = tf.reshape(out, shape=[-1, 29 * 29])
            #print(out.shape)
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
            dc_den1 = tf.nn.relu(fullyConnected(x, output_size=1000, name='dc_g_den1'))
            dc_den2 = tf.nn.relu(fullyConnected(dc_den1, output_size= 1000, name='dc_g_den2'))
            output = fullyConnected(dc_den2, output_size= 1, name='dc_g_output')
            return output

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
            dc_den1 = tf.nn.relu(fullyConnected(x, output_size=1000, name='dc_c_den1'))
            dc_den2 = tf.nn.relu(fullyConnected(dc_den1, output_size=1000, name='dc_c_den2'))
            output = fullyConnected(dc_den2, output_size=1, name='dc_c_output')
            return output