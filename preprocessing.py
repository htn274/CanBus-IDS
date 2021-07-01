import pandas as pd
import numpy as np
import glob
import dask.dataframe as dd
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import argparse

def fill_flag(sample):
    if not isinstance(sample['Flag'], str):
        col = 'Data' + str(sample['DLC'])
        sample['Flag'] = sample[col]
    return sample

def convert_canid_bits(cid):
    try:
        s = bin(int(str(cid), 16))[2:].zfill(29)
        bits = list(map(int, list(s)))
        return bits
    except:
        return None
    
def preprocess(file_name):
    df = dd.read_csv(file_name, header=None, names=attributes)
    print('Reading from {}: DONE'.format(file_name))
    print('Dask processing: -------------')
    df = df.apply(fill_flag, axis=1)
    pd_df = df.compute()
    pd_df = pd_df[['Timestamp', 'canID', 'Flag']]
    pd_df['canBits'] = pd_df.canID.apply(convert_canid_bits)
    print('Dask processing: DONE')
    print('Aggregate data -----------------')
    as_strided = np.lib.stride_tricks.as_strided  
    test_df = pd_df.reset_index()
    win = 29
    v = as_strided(test_df.canBits, (len(test_df) - (win - 1), win), (test_df.canBits.values.strides * 2))
    test_df['Flag'] = test_df['Flag'].apply(lambda x: True if x == 'T' else False)
    test_df['features'] = pd.Series(v.tolist(), index=test_df.index[win - 1:])
    # test_df['features'] = test_df.features.apply(lambda x: np.array(x).ravel().tolist())
    v = as_strided(test_df.Flag, (len(test_df) - (win - 1), win), (test_df.Flag.values.strides * 2))
    test_df['label'] = pd.Series(v.tolist(), index=test_df.index[win - 1:])
    test_df = test_df.iloc[win - 1:]
    test_df['label'] = test_df['label'].apply(lambda x: 1 if any(x) else 0)
    print('Preprocessing: DONE')
    return test_df[['features', 'label']].reset_index().drop(['index'], axis=1)

def create_train_test(df):
    print('Create train - test - val: ')
    train, test = train_test_split(df, test_size=0.3, shuffle=True)
    train, val = train_test_split(train, test_size=0.2, shuffle=True)
    train_ul, train_l = train_test_split(train, test_size=0.1, shuffle=True)
    train_ul = train_ul.reset_index().drop(['index'], axis=1)
    train_l = train_l.reset_index().drop(['index'], axis=1)
    test = test.reset_index().drop(['index'], axis=1)
    val = val.reset_index().drop(['index'], axis=1)
    
    data_info = {
        "train_unlabel": train_ul.shape[0],
        "train_label": train_l.shape[0],
        "validation": val.shape[0],
        "test": test.shape[0]
    }
    
    return data_info, train_ul, train_l, val, test

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

def write_tfrecord(data, filename):
    tfrecord_writer = tf.io.TFRecordWriter(filename)
    for _, row in tqdm(data.iterrows()):
        tfrecord_writer.write(serialize_example(row['features'], row['label']))
    tfrecord_writer.close()    

def main(indir, outdir, attacks):
    attributes = ['Timestamp', 'canID', 'DLC', 
                           'Data0', 'Data1', 'Data2', 
                           'Data3', 'Data4', 'Data5', 
                           'Data6', 'Data7', 'Flag']
    attack_types = ['DoS', 'Fuzzy', 'gear', 'RPM']

    for attack in attack_types[1:]:
        file_name = '{}/{}_dataset.csv'.format(indir, attack)
        print(file_name + '---------------------------')
        df = preprocess(file_name)
        data_info, train_ul, train_l, val, test = create_train_test(df)
        save_path = '{}/{}/'.format(outdir, attack)
        print('Path: ', save_path)
        print('Writing train_unlabel.......................')
        write_tfrecord(train_ul, save_path + "train_unlabel")
        print('Writing train_label.......................')
        write_tfrecord(train_l, save_path + "train_label")
        print('Writing test.......................')
        write_tfrecord(test, save_path + "test")
        print('Writing val.......................')
        write_tfrecord(val, save_path + "val")
        print('Writing data info')
        json.dump(data_info, open(save_path + 'datainfo.txt', 'w'))
        print('==========================================')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="./Data/Car-Hacking")
    parser.add_argument('--outdir', type=str, default="./Data/")
    parser.add_argument('--attack_type', type=str, default="all", nargs='+')
    args = parser.parse_args()
    
    if args.attack_type == 'all':
        attack_types = ['DoS', 'Fuzzy', 'gear', 'RPM']
    else:
        attack_types = args.attack_type
     
    main(args.indir, args.outdir, args.attack_type)
        
    
    