import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import json
import glob
import tqdm
from utils import *
from AAE import AAE
from CAAE import CAAE
import argparse
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

class Model:
    def __init__(self, model='AAE', unknown_attack=None, input_dim=29*29, z_dim=10, batch_size=100, n_epochs=100, supervised_lr=0.0001, reconstruction_lr=0.0001, regularization_lr=0.0001):
        self.unknown_attack = unknown_attack
        self.read_datainfo()
        self.input_dim = input_dim
        self.n_l1 = 1000
        self.n_l2 = 1000
        self.z_dim = z_dim
        self.batch_size_unknown = 0
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        # learning_rate = 0.001
        self.supervised_lr = supervised_lr
        self.reconstruction_lr = reconstruction_lr
        self.regularization_lr = regularization_lr
        self.beta1 = 0.9
        self.n_labels = 2
        self.n_labeled = self.data_info['train_label']
        self.validation_size = self.data_info['validation']
        if model == 'AAE':
            self.model = AAE(self.input_dim, self.n_l1, self.n_l2, self.z_dim, self.n_labels)
            self.model_name = ''
        else:
            self.model = CAAE(self.n_labels, self.z_dim)
            self.model_name = 'CNN'
        
    def read_datainfo(self):
        self.data_info = {
       "train_unlabel": 0, 
        "train_label": 0, 
        "validation": 0, 
        "test": 0
        }
        self.labels = ['DoS_50', 'Fuzzy', 'gear', 'RPM', 'Normal']
        for f in ['./Data/{}/datainfo.txt'.format(l) for l in self.labels if l != self.unknown_attack]:
            data_read = json.load(open(f))
            for key in self.data_info.keys():
                self.data_info[key] += data_read[key]

        self.attack = 'all' # DoS, Fuzzy, gear, RPM, all
        if self.unknown_attack != None:
            self.results_path = './Results/unknown/{}'.format(self.unknown_attack)
        else:
            self.results_path = './Results/{}/'.format(self.attack)
        print('Unknown attack: {}'.format(self.unknown_attack if self.unknown_attack!='' else 'None'))
        print('Data info: ', self.data_info)
        
    def construct_data_flow(self):
        train_unlabel_paths = ['./Data/{}/train_unlabel'.format(l) for l in self.labels]
        #unknown_train_unlabel_path = ['./Data/{}/train_unlabel'.format(self.unknown_attack)]
        train_label_paths = ['./Data/{}/train_label'.format(l) for l in self.labels if l != self.unknown_attack]
        val_paths = ['./Data/{}/val'.format(l) for l in self.labels if l != self.unknown_attack]
        
        print('Unlabeled data: ', train_unlabel_paths)
        print('Label data:', train_label_paths)
        
        train_unlabel = data_from_tfrecord(train_unlabel_paths, self.batch_size - self.batch_size_unknown, self.n_epochs)
        # train_unlabel_unknown = data_from_tfrecord(unknown_train_unlabel_path, self.batch_size_unknown, self.n_epochs)
        train_label = data_from_tfrecord(train_label_paths, self.batch_size, self.n_epochs)
        validation = data_from_tfrecord(val_paths, self.batch_size, self.n_epochs)
        
        if self.unknown_attack != None:
            val_unknown_path = ['./Data/{}/val'.format(a) for a in [self.unknown_attack, 'Normal']]
            data_info_unknown_attack = json.load(open('./Data/{}/datainfo.txt'.format(self.unknown_attack)))
            self.validation_unknown_size = data_info_unknown_attack['validation']
            self.validation_unknown = data_from_tfrecord(val_unknown_path, self.batch_size, self.n_epochs) 
        
        return train_unlabel, train_label, validation
        
    
    def build(self):
        #Define place holder
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_dim], name='Input')
        self.x_input_l = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_dim], name='Labeled_Input')
        self.y_input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.n_labels], name='Labels')
        self.x_target = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_dim], name='Target')
        self.real_distribution = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.z_dim], name='Real_distribution')
        self.categorial_distribution = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.n_labels],
                                                 name='Categorical_distribution')
        self.manual_decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, self.z_dim + self.n_labels], name='Decoder_input')
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        # Reconstruction Phase
        # Encoder try to predict both label and latent space of the input, which will be feed into Decoder to reconstruct the input
        # The process is optimized by autoencoder_loss which is the MSE of the decoder_output and the orginal input
        with (tf.variable_scope(tf.get_variable_scope())):
            self.encoder_output_label, self.encoder_output_latent = self.model.encoder(self.x_input)
            decoder_input = tf.concat([self.encoder_output_label, self.encoder_output_latent], 1)
            decoder_output = self.model.decoder(decoder_input)

        self.autoencoder_loss = tf.reduce_mean(tf.square(self.x_target - decoder_output))
        self.autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(self.autoencoder_loss)
        # Regularization Phase
        # Train both 2 discriminator of gaussian and categorical to detect the output from encoder
        with (tf.variable_scope(tf.get_variable_scope())):
            # Discriminator for gaussian
            d_g_real = self.model.discriminator_gauss(self.real_distribution)
            d_g_fake = self.model.discriminator_gauss(self.encoder_output_latent, reuse=True)
        # Need to seperate dicriminator of gaussian and categorical
        with (tf.variable_scope(tf.get_variable_scope())):
            # Discrimnator for categorical
            d_c_real = self.model.discriminator_categorical(self.categorial_distribution)
            d_c_fake = self.model.discriminator_categorical(self.encoder_output_label, reuse=True)

        # Discriminator gaussian loss 
        dc_g_loss_real = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_g_real), logits=d_g_real))
        dc_g_loss_fake = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_g_fake), logits=d_g_fake))
        self.dc_g_loss = dc_g_loss_real + dc_g_loss_fake

        # Discriminator categorical loss
        dc_c_loss_real = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_c_real), logits=d_c_real))
        dc_c_loss_fake = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_c_fake), logits=d_c_fake))
        self.dc_c_loss = dc_c_loss_fake + dc_c_loss_real

        all_variables = tf.trainable_variables()
        dc_g_var = [var for var in all_variables if 'dc_g_' in var.name]
        dc_c_var = [var for var in all_variables if 'dc_c_' in var.name]
        self.discriminator_g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                               beta1=self.beta1).minimize(self.dc_g_loss, var_list=dc_g_var)
        self.discriminator_c_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                               beta1=self.beta1).minimize(self.dc_c_loss, var_list=dc_c_var)
        # Generator loss
        generator_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_g_fake), logits=d_g_fake))
        generator_c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_c_fake), logits=d_c_fake))
        self.generator_loss = generator_g_loss + generator_c_loss

        en_var = [var for var in all_variables if 'e_' in var.name]
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(self.generator_loss, var_list=en_var)
        
        # Semi-Supervised Classification Phase
        # Train encoder with a small amount of label samples
        with tf.variable_scope(tf.get_variable_scope()):
            self.encoder_output_label_, self.encoder_output_latent_ = self.model.encoder(self.x_input_l, reuse=True, supervised=True)

        # Classification accuracy of encoder
        self.output_label = tf.argmax(self.encoder_output_label_, 1)
        correct_pred = tf.equal(self.output_label, tf.argmax(self.y_input, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.supervised_encoder_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.encoder_output_label_))
        self.supervised_encoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(self.supervised_encoder_loss, var_list=en_var)

        

    def get_val_acc(self, val_size, batch_size, tfdata, sess):
        acc = 0
        y_true, y_pred = [], []
        num_batches = int(val_size/batch_size)
        for j in tqdm.tqdm(range(num_batches)):
            batch_x_l, batch_y_l = data_stream(tfdata, sess)
            batch_pred = sess.run(self.output_label, feed_dict={self.x_input_l: batch_x_l, self.y_input: batch_y_l})

            batch_label = np.argmax(batch_y_l, axis=1)
            y_pred += batch_pred.tolist()
            y_true += batch_label.tolist()

        avg_acc = np.equal(y_true, y_pred).mean()

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fnr = fn/(tp + fn)
        err = (fn + fp) / (tp + tn + fp + fn)
        precision = tp/(tp + fp)
        recall = 1 - fnr
        f1 = (2 * precision * recall) / (precision + recall)

        return avg_acc, precision, recall, f1
        
    def train(self):
        train_unlabel, train_label, validation = self.construct_data_flow()
        self.build()
        init = tf.global_variables_initializer()
        # Tensorboard visualization
        tf.summary.scalar(name='Autoencoder Loss', tensor=self.autoencoder_loss)
        tf.summary.scalar(name='Discriminator gauss Loss', tensor=self.dc_g_loss)
        tf.summary.scalar(name='Discriminator categorical Loss', tensor=self.dc_c_loss)
        tf.summary.scalar(name='Generator Loss', tensor=self.generator_loss)
        tf.summary.scalar(name='Supervised Encoder Loss', tensor=self.supervised_encoder_loss)
        # tf.summary.scalar(name='Supervised Encoder Accuracy', tensor=accuracy)
        tf.summary.histogram(name='Encoder Gauss Distribution', values=self.encoder_output_latent)
        tf.summary.histogram(name='Real Gauss Distribution', values=self.real_distribution)
        tf.summary.histogram(name='Encoder Categorical Distribution', values=self.encoder_output_label)
        tf.summary.histogram(name='Real Categorical Distribution', values=self.categorial_distribution)
        self.summary_op = tf.summary.merge_all()
        accuracies = []
        # Saving the model
        saver = tf.train.Saver()
        step = 0
        # Early stopping
        save_sess = None
        best_acc = 1
        stop = False
        last_improvement = 0
        require_improvement = 20
        

        with tf.Session() as sess:
            tensorboard_path, saved_model_path, log_path = form_results(self.model_name, self.results_path, self.z_dim, self.supervised_lr, self.batch_size, self.n_epochs, self.beta1)
            sess.run(init)
            writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
            for epoch in range(self.n_epochs):
                if epoch == 50:
                    self.supervised_lr /= 10
                    self.reconstruction_lr /= 10
                    self.regularization_lr /= 10
                n_batches = int(self.n_labeled / self.batch_size)
                num_normal = 0 
                num_attack = 0
                print("------------------Epoch {}/{}------------------".format(epoch, self.n_epochs))
                for b in tqdm.tqdm(range(1, n_batches + 1)):
                    z_real_dist = np.random.randn(self.batch_size, self.z_dim) * 5.
                    real_cat_dist = np.random.randint(low=0, high=2, size=self.batch_size)
                    real_cat_dist = np.eye(self.n_labels)[real_cat_dist]

                    batch_x_ul, batch_y_ul = data_stream(train_unlabel, sess)
                    batch_x_l, batch_y_l = data_stream(train_label, sess)
                    #print(batch_x_ul)
#                     if self.unknown_attack != '':
#                         hint_x, hint_y = data_stream(train_unlabel_unknown, sess)
#                         batch_x_ul = np.append(batch_x_ul, hint_x, axis=0)
#                         batch_y_ul = np.append(batch_y_ul, hint_y, axis=0)
#                         np.random.shuffle(batch_x_ul)

                    #print(batch_x_ul)
                    num_normal += (np.argmax(batch_y_ul, axis=1) == 0).sum()
                    num_attack += (np.argmax(batch_y_ul, axis=1) == 1).sum()

    #                 print('Num normal: ', (np.argmax(batch_y_l, axis=1) == 0).sum())
    #                 print('Num attack: ', (np.argmax(batch_y_l, axis=1) == 1).sum())

                    sess.run(self.autoencoder_optimizer, feed_dict={self.x_input: batch_x_ul, self.x_target: batch_x_ul, self.learning_rate: self.reconstruction_lr})
                    sess.run(self.discriminator_g_optimizer,
                             feed_dict={self.x_input: batch_x_ul, self.x_target: batch_x_ul, self.real_distribution: z_real_dist, self.learning_rate: self.regularization_lr})
                    sess.run(self.discriminator_c_optimizer,
                             feed_dict={self.x_input: batch_x_ul, self.x_target: batch_x_ul,
                                        self.categorial_distribution: real_cat_dist, self.learning_rate: self.regularization_lr})
                    sess.run(self.generator_optimizer, feed_dict={self.x_input: batch_x_ul, self.x_target: batch_x_ul, self.learning_rate: self.regularization_lr})


                    sess.run(self.supervised_encoder_optimizer, feed_dict={self.x_input_l: batch_x_l, self.y_input: batch_y_l, self.learning_rate: self.supervised_lr})
                    if b % 10 == 0:
                        a_loss, d_g_loss, d_c_loss, g_loss, s_loss, summary = sess.run(
                            [self.autoencoder_loss, self.dc_g_loss, self.dc_c_loss, self.generator_loss, self.supervised_encoder_loss,
                             self.summary_op],
                            feed_dict={self.x_input: batch_x_ul, self.x_target: batch_x_ul,
                                       self.real_distribution: z_real_dist, self.y_input: batch_y_l, self.x_input_l: batch_x_l,
                                       self.categorial_distribution: real_cat_dist})
                        writer.add_summary(summary, global_step=step)
                        with open(log_path + '/log.txt', 'a') as log:
                            log.write("Epoch: {}, iteration: {}\n".format(epoch, b))
                            log.write("Autoencoder Loss: {}\n".format(a_loss))
                            log.write("Discriminator Gauss Loss: {}".format(d_g_loss))
                            log.write("Discriminator Categorical Loss: {}".format(d_c_loss))
                            log.write("Generator Loss: {}\n".format(g_loss))
                            log.write("Supervised Loss: {}".format(s_loss))
                    step += 1

                print('Num normal: ', num_normal)
                print('Num attack: ', num_attack)

                if (epoch + 1) % 5 == 0:
                    print("Runing on validation...----------------")
                    acc_known, precision_known, recall_known, f1_known = self.get_val_acc(self.validation_size, self.batch_size, validation, sess)
                    print("Accuracy on Known attack: {}".format(acc_known))
                    print("Precision on Known attack: {}".format(precision_known))
                    print("Recall on Known attack: {}".format(recall_known))
                    print("F1 on Known attack: {}".format(f1_known))
                    
                    acc_unknown, precision_unknown, recall_unknown, f1_unknown = self.get_val_acc(self.validation_unknown_size, self.batch_size, self.validation_unknown, sess)
                    print("Accuracy on unKnown attack: {}".format(acc_unknown))
                    print("Precision on unKnown attack: {}".format(precision_unknown))
                    print("Recall on unKnown attack: {}".format(recall_unknown))
                    print("F1 on unKnown attack: {}".format(f1_unknown))
                    
                    print('Save model')
                    saver.save(sess, save_path=saved_model_path, global_step=step)
                    

    def test(self, results_path, unknown_test = False):
        self.build()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        if unknown_test:
            data_path = ['./Data/{}/'.format(a) for a in [self.unknown_attack, 'Normal']]
            test_size = 0
            for f in ['{}/datainfo.txt'.format(p) for p in data_path]:
                data_read = json.load(open(f))
                test_size += data_read['test']
        else:
            test_size = self.data_info['test']
            data_path = ['./Data/{}/'.format(a) for a in self.labels]
        # results_path = './Results/all/CNN_2021-07-21 19:53:22.883136_10_0.0001_64_300_0.9_Semi_Supervised/'
        print('Test data: ', data_path)
        with tf.Session() as sess:
            saver.restore(sess, save_path=tf.train.latest_checkpoint(results_path + '/Saved_models'))
            test = data_from_tfrecord([p + 'test' for p in data_path], self.batch_size, 1)
            num_batches = int(test_size / self.batch_size)
            y_true = np.empty((0), int)
            y_pred = np.empty((0), int)
            total_prob = np.empty((0), float)
            total_latent = np.empty((0, self.z_dim), float)
            
            for _ in tqdm.tqdm(range(num_batches)):
                x_test, y_test = data_stream(test, sess)
                batch_pred, batch_latent = sess.run([self.output_label, self.encoder_output_latent_], feed_dict={self.x_input_l: x_test})
                total_latent = np.append(total_latent, batch_latent, axis=0)
                batch_label = np.argmax(y_test, axis=1).reshape((self.batch_size))
                #prob = np.max(batch_pred, axis=1).reshape((self.batch_size))
                #batch_pred = np.argmax(batch_pred, axis=1).reshape((self.batch_size))
                y_pred = np.append(y_pred, batch_pred, axis=0)
                y_true = np.append(y_true, batch_label, axis=0) 
                #total_prob = np.append(total_prob, prob, axis=0)
                
        evaluate(y_true, y_pred)
                    
if __name__ == '__main__':
    #python3 train.py --unknown_attack 'DoS' --model "CAAE" --batch_size 64 --is_train
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str, default=None)
    parser.add_argument('--model', type=str, default="CAAE")
    parser.add_argument('--unknown_attack', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--is_train', action='store_true')
    args = parser.parse_args()
    
    model = Model(model=args.model, unknown_attack = args.unknown_attack, batch_size=args.batch_size, n_epochs=args.epochs)
    if args.is_train:
        model.train()
    else:
        if args.res_path is None:
            print("Must define res_path which store model's weights")
        else:
            #res_path = './Results/all/CNN_2021-07-21 19:53:22.883136_10_0.0001_64_300_0.9_Semi_Supervised/'
            #res_path = './Results/unknown/DoS/2021-07-21 15:02:31.836424_10_0.0001_100_300_0.9_Semi_Supervised/'
            model.test(args.res_path, unknown_test=True)
