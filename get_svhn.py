# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:20:46 2017

@author: RMFit
"""
# Imports
from __future__ import print_function
import digitStruct
import os
import cv2
import numpy as np
import scipy.io
import sys
import tarfile
from six.moves.urllib.request import urlretrieve

np.random.seed(1)

def maybe_download(url, filename, expected_bytes, force=False):
      """Download a file if not present, and make sure it's the right size."""
      if force or not os.path.exists(filename):
        print('Attempting to download:', filename) 
        filename, _ = urlretrieve(url + filename, filename)
        print('\nDownload Complete!')
      statinfo = os.stat(filename)
      if statinfo.st_size == expected_bytes:
        pass
      else:
        raise Exception(
          'Failed to verify ' + filename + '. Can you get to it with a browser?')
      return filename

def maybe_extract(filename, force=False):
      root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
      if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        pass
      else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
      data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d)) and (len(d) == 1)]   

def read_mat_7_3(mat_file):
    objectList = []
    x_pix = []
    y_pix = []
    for dsObj in digitStruct.yieldNextDigitStruct(mat_file):
        label = ''
        bounding = []
        for bbox in dsObj.bboxList:
            label += str(bbox.label)
            boundBox = (bbox.label, bbox.left, bbox.top, bbox.width, bbox.height)
            bounding.append(boundBox)
        try:
          image_name = mat_file.split('\\')[0] + '\\' + dsObj.name
          image = cv2.imread(image_name, 0)
          if isinstance(image, np.ndarray):
              y = len(image)
              x = len(image[0])
              x_pix.append(x)
              y_pix.append(y)
              data = (image_name, x, y, bounding, label)              
              objectList.append(data)
        except IOError as e:
          print('Could not read:', image_name, ':', e, '- it\'s ok, skipping.')
    data_len = len(objectList)
    x = max(x_pix)
    y = max(y_pix)
    print(data_len, x, y)
    dataset = np.ndarray((data_len, 2), dtype='|S16')
    bbox_set = np.ndarray((data_len, 6, 5), dtype=np.int16)
    sizes = np.ndarray((data_len, 2), dtype=np.int16)
    for s, sample in enumerate(objectList):
        dataset[s, 0] = sample[0]
        dataset[s, 1] = sample[4]
        sizes[s, 0] = sample[1]
        sizes[s, 1] = sample[2]
        for b, bbox in enumerate(sample[3]):
            bbox_set[s, b, :] = bbox
    return dataset, bbox_set, sizes
    
def get_mat_7_3(mat_file, force=False):
    filename = mat_file[:-4] + '.npz'
    force=True
    if force or not os.path.exists(filename):
        print('Attempting to build:', filename) 
        dataset, bbox, img_dims = read_mat_7_3(mat_file)
        data = {'dataset':dataset, 'bbox':bbox, 'img_dims':img_dims}
        print('\nBuild Complete!')
        np.savez(filename, **data)
    else:
        data = np.load(filename)
        
    return data['dataset'], data['bbox'], data['img_dims']
    
      
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

def big_svhn_dataset():
   
    svhn_url = 'http://ufldl.stanford.edu/housenumbers/'

    print('Starting')
    train_filename = maybe_download(svhn_url, 'train.tar.gz', 404141560)
    test_filename = maybe_download(svhn_url, 'test.tar.gz', 276555967)
    extra_filename = maybe_download(svhn_url, 'extra.tar.gz', 1955489752)
    print('Download Complete')

    maybe_extract(train_filename)
    maybe_extract(test_filename)
    maybe_extract(extra_filename)
    print('Extract Complete')

    test_dataset, test_bbox, test_labels = get_mat_7_3("test\digitStruct.mat")
    train_dataset, train_bbox, train_labels = get_mat_7_3("train\digitStruct.mat")
    extra_dataset, extra_bbox, extra_labels = get_mat_7_3("extra\digitStruct.mat")
    print('Loading Complete')

    dataset = {}
    dataset['train_dataset'] = extra_dataset[32000:]
    dataset['train_labels'] = extra_labels[32000:]
    dataset['valid_dataset'] = extra_dataset[:32000]
    dataset['valid_labels'] = extra_labels[:32000]
    dataset['test_dataset'] = test_dataset
    dataset['test_labels'] = test_labels
    print('Dataset Built')    

    return dataset

def make_composite_dataset(size):
    if (size == 'big'):
        return big_svhn_dataset()
    else:
        return small_svhn_dataset()
    
svhn_data = make_composite_dataset('big')
#import matplotlib.pyplot as plt
#plt.imshow(svhn_data['train_dataset'][0])
#print(svhn_data['train_labels'][0])
