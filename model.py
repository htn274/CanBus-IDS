import tensorflow as tf
import numpy as np

# input_dim = 29 * 29
# n_l1 = 1000
# n_l2 = 1000
# z_dim = 10
# n_labels = 2

class AAE:
    def __init__(self, input_dim, n_l1, n_l2, z_dim, n_labels):
        self.input_dim = input_dim
        self.n_l1 = n_l1
        self.n_l2 = n_l2
        self.z_dim = z_dim
        self.n_labels = n_labels

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

    def encoder(self, x, reuse=False, supervised=False):
        """
        Encode part of the autoencoder.
        :param x: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :param supervised: True -> returns output without passing it through softmax,
                           False -> returns output after passing it through softmax.
        :return: tensor which is the classification output and a hidden latent variable of the autoencoder.
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Encoder'):
            e_dense_1 = tf.nn.relu(self.dense(x, self.input_dim, self.n_l1, 'e_dense_1'))
            e_dense_2 = tf.nn.relu(self.dense(e_dense_1, self.n_l1, self.n_l2, 'e_dense_2'))
            latent_variable = self.dense(e_dense_2, self.n_l2, self.z_dim, 'e_latent_variable')
            cat_op = self.dense(e_dense_2, self.n_l2, self.n_labels, 'e_label')
            if not supervised:
                softmax_label = tf.nn.softmax(logits=cat_op, name='e_softmax_label')
            else:
                softmax_label = cat_op
            return softmax_label, latent_variable

    def decoder(self, x, reuse=False):
        """
        Decoder part of the autoencoder.
        :param x: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Decoder'):
            d_dense_1 = tf.nn.relu(self.dense(x, self.z_dim + self.n_labels, self.n_l2, 'd_dense_1'))
            d_dense_2 = tf.nn.relu(self.dense(d_dense_1, self.n_l2, self.n_l1, 'd_dense_2'))
            output = tf.nn.sigmoid(self.dense(d_dense_2, self.n_l1, self.input_dim, 'd_output'))
            return output

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
            dc_den1 = tf.nn.relu(self.dense(x, self.z_dim, self.n_l1, name='dc_g_den1'))
            dc_den2 = tf.nn.relu(self.dense(dc_den1, self.n_l1, self.n_l2, name='dc_g_den2'))
            output = self.dense(dc_den2, self.n_l2, 1, name='dc_g_output')
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
            dc_den1 = tf.nn.relu(self.dense(x, self.n_labels, self.n_l1, name='dc_c_den1'))
            dc_den2 = tf.nn.relu(self.dense(dc_den1, self.n_l1, self.n_l2, name='dc_c_den2'))
            output = self.dense(dc_den2, self.n_l2, 1, name='dc_c_output')
            return output