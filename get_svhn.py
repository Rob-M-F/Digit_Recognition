# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:20:46 2017

@author: RMFit
"""
# Imports
import os
import numpy as np
import scipy.io
from six.moves.urllib.request import urlretrieve

np.random.seed(1)

def maybe_download(url, filename, expected_bytes, force=False):
      """Download a file if not present, and make sure it's the right size."""
      if force or not os.path.exists(filename):
        print('Attempting to download:', filename) 
        filename, _ = urlretrieve(url + filename, filename)
        print('\nDownload Complete!')
      statinfo = os.stat(filename)
      print(statinfo.st_size)
      if statinfo.st_size == expected_bytes:
        pass
      else:
        raise Exception(
          'Failed to verify ' + filename + '. Can you get to it with a browser?')
      return filename
      
def get_svhn_data_labels(dataset):
    working_data = np.swapaxes(dataset['X'], 2, 3)
    working_data = np.swapaxes(working_data, 1, 2)
    working_data = np.swapaxes(working_data, 0, 1)
    working_labels = dataset['y']
    return working_data, working_labels

def small_svhn_dataset():
    
    svhn_url = 'http://ufldl.stanford.edu/housenumbers/'

    print('Starting')
    train_filename = maybe_download(svhn_url, 'train_32x32.mat', 182040794)
    test_filename = maybe_download(svhn_url, 'test_32x32.mat', 64275384)
    extra_filename = maybe_download(svhn_url, 'extra_32x32.mat', 1329278602)
    print('Download Complete')

    train_dataset = scipy.io.loadmat(train_filename)
    test_dataset = scipy.io.loadmat(test_filename)
    extra_dataset = scipy.io.loadmat(extra_filename)
    print('Loading Complete')
   
    train_dataset, train_labels = get_svhn_data_labels(train_dataset)
    extra_dataset, extra_labels = get_svhn_data_labels(extra_dataset)
    extra_dataset = np.append(extra_dataset, train_dataset, axis=0)
    extra_labels = np.append(extra_labels, train_labels, axis=0)
    test_dataset, test_labels = get_svhn_data_labels(test_dataset)
    print('Prepared')  
    
    dataset = {}
    dataset['train_dataset'] = extra_dataset[32000:]
    dataset['train_labels'] = extra_labels[32000:]
    dataset['valid_dataset'] = extra_dataset[:32000]
    dataset['valid_labels'] = extra_labels[:32000]
    dataset['test_dataset'] = test_dataset
    dataset['test_labels'] = test_labels
    print('Dataset Built')    

    return dataset

def make_composite_dataset():
    return small_svhn_dataset()
    
svhn_data = make_composite_dataset()
#import matplotlib.pyplot as plt
#plt.imshow(svhn_data['train_dataset'][0])
#print(svhn_data['train_labels'][0])