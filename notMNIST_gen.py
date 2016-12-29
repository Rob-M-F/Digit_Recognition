# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 09:53:22 2016

@author: RMFit
"""

# Imports
import os
import sys
import random
import tarfile
import numpy as np
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
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
  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))
  return data_folders
  

image_size = 28  # Pixel height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  dataset_length = 53000
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if force or not os.path.exists(set_filename):
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      if dataset.shape[0] > dataset_length:
          dataset_length = dataset.shape[0]
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  return dataset_names

def gen_data_dict(pickle_files):
  data_dict = {}
  for pickle_file in pickle_files: 
    try:
      with open(pickle_file, 'rb') as f:
        data_dict[pickle_file[-8:-7]] = pickle.load(f)[:]
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
  return data_dict

def gen_dataset(sourceDict, dataSamples=10000, minDigits=3, maxDigits=5):
    dataset = []
    labels = []
    for i in range(dataSamples):
        sampleLen = random.randint(minDigits, maxDigits)
        sampleNum = []
        sampleLabel = ''
        for j in range(sampleLen):
            letter = random.choice(sourceDict.keys())
            image = np.array(random.choice(sourceDict[letter]))
#            print(image.shape)
            sampleNum.append(image)
            sampleLabel += letter
        for j in range(sampleLen, maxDigits):
#            sampleNum.append(np.zeros((28, 28)))
#            sampleNum.append(np.ones((28, 28)))
            sampleNum.append(np.random.randn(28, 28))
            sampleLabel += '_'
        
        dataset.append(np.hstack(sampleNum))
        labels.append(sampleLabel)

    dataset = np.asarray(dataset)    
    labels = np.asarray(labels)    
    return dataset, labels
    
  

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes = 10
np.random.seed(133)

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
 
train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

train_image_data = gen_data_dict(train_datasets)
test_image_data = gen_data_dict(test_datasets)

def gen_composite(train_data = train_image_data, test_data = test_image_data):
    train_dataset, train_labels = gen_dataset(train_data, 200000)
    valid_dataset, valid_labels = gen_dataset(train_data)
    test_dataset, test_labels = gen_dataset(test_data)
    
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
    
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = gen_composite()

import matplotlib.pyplot as plt
plt.imshow(train_dataset[0])

pickle_file = 'generatedData.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise