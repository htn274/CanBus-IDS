import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import os
from sklearn.metrics import confusion_matrix

def read_tfrecord(example):
    input_dim = 29 * 29
    feature_description = {
    'input_features': tf.io.FixedLenFeature([input_dim], tf.int64),
    'label': tf.io.FixedLenFeature([1], tf.int64)
    }
    return tf.io.parse_single_example(example, feature_description)

def data_from_tfrecord(tf_filepath, batch_size, repeat_time, shuffle=True):
    data = tf.data.Dataset.from_tensor_slices(tf_filepath)
    data = data.interleave(lambda x: tf.data.TFRecordDataset(x),cycle_length=len(tf_filepath), block_length=10000)
    if shuffle:
        data = data.shuffle(100000, reshuffle_each_iteration=True)
    data = data.map(read_tfrecord, num_parallel_calls=64)
    data = data.repeat(repeat_time + 1)
    data = data.batch(batch_size)
    data = data.prefetch(1)
    iterator = data.make_one_shot_iterator()
    return iterator.get_next()

def data_stream(data_tf, sess):
    n_labels = 2
    data = sess.run(data_tf)
    x, y = data['input_features'], data['label']
    size = x.shape[0]
    y_one_hot = np.eye(n_labels)[y].reshape([size, n_labels])
    return x, y_one_hot

def form_results(model_name, results_path, z_dim, supervised_lr, batch_size, n_epochs, beta1):
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_{6}". \
        format(model_name, datetime.datetime.now(), z_dim, supervised_lr, batch_size, n_epochs, beta1)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path

# Note: can not change y_true and y_pred
def evaluate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn/(tp + fn)
    err = (fn + fp) / (tp + tn + fp + fn)
    precision = tp/(tp + fp)
    recall = 1 - fnr
    f1score = (2 * precision * recall) / (precision + recall)
    print(tp, fn)
    print(fp, tn)
    print('False negative rate: ', fnr)
    print('Error rate: ', err)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 score: ', f1score)