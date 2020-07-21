import numpy as np
import pandas as pd
import os
import h5py
import tensorflow as tf

def hdf5(path, data_key = "data", target_key = "target"):
    """
        loads data from hdf5: 
        - hdf5 should have 'train' and 'test' groups 
        - each group should have 'data' and 'target' dataset or spcify the key
        - flatten means to flatten images N * (C * H * W) as N * D array
    """
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get(data_key)[:]
        y_tr = train.get(target_key)[:]
        test = hf.get('test')
        X_te = test.get(data_key)[:]
        y_te = test.get(target_key)[:]
        X_tr = X_tr.reshape((X_tr.shape[0], 16, 16, 1))*255.
        X_te = X_te.reshape((X_te.shape[0], 16, 16, 1))*255.
#         y_tr = y_tr.reshape((y_tr.shape[0], 1))                    
#         y_te = y_te.reshape((y_te.shape[0], 1))                    
    return X_tr, y_tr, X_te, y_te

def get_usps_tf_dataset(path):
    """returns tf.data.Dataset object with USPS dataset.
    
    Arguments:
    path: path to h5 file."""
    
    x_train, y_train, x_test, y_test = hdf5(path)
    print(x_train.shape, y_train.shape)
    x_train, y_train = tf.constant(x_train), tf.constant(y_train)
    x_test, y_test = tf.constant(x_test), tf.constant(y_test)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    return train_dataset, test_dataset

                            

