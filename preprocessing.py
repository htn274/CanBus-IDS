"""
Used to convert .csv into tfrecord format
"""
import pandas as pd
import numpy as np
import glob
import dask.dataframe as dd
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import argparse

attributes = ['Timestamp', 'canID', 'DLC', 
                           'Data0', 'Data1', 'Data2', 
                           'Data3', 'Data4', 'Data5', 
                           'Data6', 'Data7', 'Flag']

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
    pd_df = pd_df[['Timestamp', 'canID', 'Flag']][:100]
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
    data_info = {}
    for attack in attacks:
        print('Attack: {} ==============='.format(attack))
        finput = '{}/{}_dataset.csv'.format(indir, attack)
        df = preprocess(finput)
        print("Writing...................")
        foutput_attack = '{}/{}'.format(outdir, attack)
        foutput_normal = '{}/Normal_{}'.format(outdir, attack)
        df_attack = df[df['label'] == 1]
        df_normal = df[df['label'] == 0]
        write_tfrecord(df_attack, foutput_attack)
        write_tfrecord(df_normal, foutput_normal)
        
        data_info[foutput_attack] = df_attack.shape[0]
        data_info[foutput_normal] = df_normal.shape[0]
        
    json.dump(data_info, open('{}/datainfo.txt'.format(outdir), 'w'))
    print("DONE!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="./Data/Car-Hacking")
    parser.add_argument('--outdir', type=str, default="./Data/TFRecord/")
    parser.add_argument('--attack_type', type=str, default="all", nargs='+')
    args = parser.parse_args()
    
    if args.attack_type == 'all':
        attack_types = ['DoS', 'Fuzzy', 'gear', 'RPM']
    else:
        attack_types = [args.attack_type]
     
    main(args.indir, args.outdir, attack_types)
        
    
    