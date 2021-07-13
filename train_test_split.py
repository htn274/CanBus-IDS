import pandas as pd
import numpy as np
import glob
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import os
import argparse


def serialize_example(x, y):
    """converts x, y to tf.train.Example and serialize"""
    #Need to pay attention to whether it needs to be converted to numpy() form
    input_features = tf.train.Int64List(value = np.array(x).flatten())
    label = tf.train.Int64List(value = np.array([y]))
    features = tf.train.Features(
        feature = {
            "input_features": tf.train.Feature(int64_list = input_features),
            "label" : tf.train.Feature(int64_list = label)
        }
    )
    example = tf.train.Example(features = features)
    return example.SerializeToString()

def read_tfrecord(example):
    input_dim = 29 * 29
    feature_description = {
    'input_features': tf.io.FixedLenFeature([input_dim], tf.int64),
    'label': tf.io.FixedLenFeature([1], tf.int64)
    }
    return tf.io.parse_single_example(example, feature_description)

def data_from_tfrecord(tf_filepath, batch_size, repeat_time):
    data = tf.data.TFRecordDataset(tf_filepath)
    data = data.map(read_tfrecord)
    data = data.shuffle(2)
    data = data.repeat(repeat_time + 1)
    data = data.batch(batch_size)
    # print(tf.data.experimental.cardinality(data))
    iterator = data.make_one_shot_iterator()
    return iterator.get_next()

def data_helper(data_tf, sess):
    n_labels = 2
    data = sess.run(data_tf)
    x, y = data['input_features'], data['label']
    size = x.shape[0]
    y_one_hot = np.eye(n_labels)[y].reshape([size, n_labels])
    return x, y_one_hot

def write_tfrecord(data, filename):
    print('Writing {}================= '.format(filename))
    iterator = data.make_one_shot_iterator().get_next()
    init = tf.global_variables_initializer()
    tfrecord_writer = tf.io.TFRecordWriter(filename)
    with tf.Session() as sess:
        sess.run(init)
        while True:
            try:
                batch_data = sess.run(iterator)
                for x, y in zip(batch_data['input_features'], batch_data['label']):
                    tfrecord_writer.write(serialize_example(x, y))
            except:
                break
            
    tfrecord_writer.close()
    
def train_test_split(source_path, dest_path, DATASET_SIZE,\
                     train_label_size = 100 * 1000, train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15):

    train_size = int(DATASET_SIZE * train_ratio)
    val_size = int(DATASET_SIZE * val_ratio)
    test_size = int(DATASET_SIZE * test_ratio)
    print(train_size, val_size, test_size)
    dataset = tf.data.TFRecordDataset(source_path)
    dataset = dataset.map(read_tfrecord)
    dataset = dataset.shuffle(10)
    train = dataset.take(train_size)
    train_label = train.take(train_label_size)
    train_unlabel = train.skip(train_label_size)
    val = dataset.skip(train_size)
    test = val.skip(val_size)
    val = val.take(val_size)
    batch_size = 10000
    train_label = train_label.batch(batch_size)
    train_unlabel = train_unlabel.batch(batch_size)
    test = test.batch(batch_size)
    val = val.batch(batch_size)

    train_test_info = {
        "train_unlabel": train_size - train_label_size,
        "train_label": train_label_size,
        "validation": val_size,
        "test": test_size
    }
    json.dump(train_test_info, open(dest_path + 'datainfo.txt', 'w'))
    write_tfrecord(train_label, dest_path + 'train_label')
    write_tfrecord(train_unlabel, dest_path + 'train_unlabel')
    write_tfrecord(test, dest_path + 'test')
    write_tfrecord(val, dest_path + 'val')
    
def main_attack(indir, outdir, attack_types):
    data_info = json.load(open('{}/datainfo.txt'.format(indir)))
    for attack in attack_types:
        print("Attack: {} ==============".format(attack))
        source = '{}/{}'.format(indir, attack)
        dest = '{}/{}/'.format(outdir, attack)
        if not os.path.exists(dest):
            os.makedirs(dest)
        train_test_split(source, dest, data_info[source])
        
def main_normal(indir, outdir, attack_types):
    normal_size = 0
    data_info = json.load(open('{}/datainfo.txt'.format(indir)))
    for attack in attack_types:
        normal_size += data_info['{}/Normal_{}'.format(indir, attack)]
    sources = ['{}/Normal_{}'.format(indir, a) for a in attack_types]
    dest = '{}/Normal/'.format(outdir)
    if not os.path.exists(dest):
        os.makedirs(dest)
        
    train_test_split(sources, dest, normal_size, train_label_size = 400 * 1000)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="./Data/TFRecord")
    parser.add_argument('--outdir', type=str, default="./Data")
    parser.add_argument('--attack_type', type=str, nargs='+')
    parser.add_argument('--normal', type=bool, default=False)
    args = parser.parse_args()
    
    if args.attack_type[0] == 'all':
        attack_types = ['DoS', 'Fuzzy', 'gear', 'RPM']
    else:
        attack_types = args.attack_type
    
    if args.normal:
        main_normal(args.indir, args.outdir, attack_types)
    else:
        main_attack(args.indir, args.outdir, attack_types)