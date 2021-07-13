import pandas as pd
import numpy as np
import tensorflow as tf

def read_tfrecord(example):
    input_dim = 29 * 29
    feature_description = {
    'input_features': tf.io.FixedLenFeature([input_dim], tf.int64),
    'label': tf.io.FixedLenFeature([1], tf.int64)
    }
    return tf.io.parse_single_example(example, feature_description)

def data_from_tfrecord(tf_filepath, batch_size, repeat_time):
    data = tf.data.Dataset.from_tensor_slices(tf_filepath)
    data = data.interleave(lambda x: tf.data.TFRecordDataset(x),cycle_length=len(tf_filepath), block_length=10000)
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