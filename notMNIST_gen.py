# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 09:53:22 2016

@author: RMFit
"""
def make_composite_dataset():
    # Imports
    import os
    import cv2
    import sys
    import random
    import tarfile
    import numpy as np
    from six.moves.urllib.request import urlretrieve
    
    url = 'http://commondatastorage.googleapis.com/books1000/'
    num_classes = 10
    image_size = 28  # Pixel height.
    np.random.seed(133)
            
    def maybe_download(filename, expected_bytes, force=False):
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
      if len(data_folders) != num_classes:
        raise Exception(
          'Expected %d folders, one per class. Found %d instead.' % (
            num_classes, len(data_folders)))
      return data_folders
    
    def load_letter(folder, min_num_images):
      """Load the data for a single letter label."""
      image_files = os.listdir(folder)
      dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                             dtype=np.float32)
      num_images = 0
      for image in image_files:
        image_file = os.path.join(folder, image)
        try:
          image_data = cv2.imread(image_file, 0)
          #cv2.imshow('image', image_data)
          #image_data = (ndimage.imread(image_file).astype(float) - 
          #              pixel_depth / 2) / pixel_depth
          if isinstance(image_data, np.ndarray):
              if (image_data.shape != (image_size, image_size)):
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
    
    def maybe_savez(data_folders, min_num_images_per_class, force=False):
      dataset_name = data_folders[0][:-1]+'images.npz'
      dataset = {}
      if force or not os.path.exists(dataset_name):
          for folder in data_folders:
              dataset[folder[-1:]]= load_letter(folder, min_num_images_per_class)
              print folder[-1],
          try:
            np.savez(dataset_name, **dataset)
          except Exception as e:
            print('Unable to save data to', dataset_name, ':', e)
      return dataset_name
      
    def gen_data_dict(dataset):
      data_dict = {}
      all_data = np.load(dataset)
      for letter in all_data.files: 
        try:
          data_dict[letter] = all_data[letter]
        except Exception as e:
          print('Unable to process data from', dataset, ':', e)
          raise
      all_data.close()
      return data_dict
    
    def gen_dataset(sourceDict, dataSamples=10000, minDigits=3, maxDigits=5):
        dataset = []
        labels = []
        for i in range(dataSamples):
            sampleLen = random.randint(minDigits, maxDigits)
            sampleNum = np.random.randn(image_size, 4)
            sampleLabel = ''
            for j in range(sampleLen):
                letter = random.choice(sourceDict.keys())
                image = np.array(random.choice(sourceDict[letter]))
                sampleNum = np.hstack((sampleNum, image, np.random.randn(image_size, 4)))
                sampleLabel += letter
            for j in range(sampleLen, maxDigits):
    #            sampleNum.append(np.zeros((28, 28)))
    #            sampleNum.append(np.ones((28, 28)))
                sampleNum = np.hstack((sampleNum, np.random.randn(image_size, image_size+4)))
                sampleLabel += '_'
            print(sampleNum.shape)
            sampleNum = np.vstack((np.random.randn(4, 164), sampleNum, np.random.randn(4, 164)))
            dataset.append(sampleNum)
            labels.append(sampleLabel)
    
        dataset = np.asarray(dataset)    
        labels = np.asarray(labels)    
        return dataset, labels
        
      
    print('Starting')
    train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
    print('Download Complete')
    
    
    train_folders = maybe_extract(train_filename)
    test_folders = maybe_extract(test_filename)
    print('Extract Complete')

    train_datasets = maybe_savez(train_folders, 45000)
    test_datasets = maybe_savez(test_folders, 1800)
    print('Saving Complete')
    
    
    train_image_data = gen_data_dict(train_datasets)
    test_image_data = gen_data_dict(test_datasets)
    print('Data Dictionaries Built')
    
    def gen_composite(train_data = train_image_data, test_data = test_image_data, force = False):
        dataset_name = 'notMNIST_ML_data.npz'
        if force or not os.path.exists(dataset_name):
            train_dataset, train_labels = gen_dataset(train_data, 200000)
            valid_dataset, valid_labels = gen_dataset(train_data)
            test_dataset, test_labels = gen_dataset(test_data)
            dataset = {'train_dataset':train_dataset, 'train_labels':train_labels,
                       'valid_dataset':valid_dataset, 'valid_labels':valid_labels,
                       'test_dataset':test_dataset, 'test_labels':test_labels}
            try:
                np.savez(dataset_name, **dataset)
            except Exception as e:
                print('Unable to save data to', dataset_name, ':', e)
        try: 
            dataset = np.load(dataset_name)
            train_dataset = dataset['train_dataset']
            train_labels = dataset['train_labels']
            valid_dataset = dataset['valid_dataset']
            valid_labels = dataset['valid_labels']
            test_dataset = dataset['test_dataset']
            test_labels = dataset['test_labels']
            dataset.close()
        except Exception as e:
          print('Unable to process data from', dataset, ':', e)
          raise
        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
    
    
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = gen_composite()
    
    import matplotlib.pyplot as plt
    plt.imshow(train_dataset[0])
    return gen_dataset(test_image_data)

bufferData, bufferLabels = make_composite_dataset()
import matplotlib.pyplot as plt
plt.imshow(bufferData[0])

