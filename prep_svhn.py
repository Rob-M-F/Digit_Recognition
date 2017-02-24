# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:36:24 2017

@author: RMFit
"""
import os
import cv2
import get_svhn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_image(image_dir):
    image = cv2.imread(image_dir.decode('utf-8'), 1)
    image = image.astype(np.float32)
#    image = (image.astype(np.float32) - 127.) / 127.
    return image

def make_window(image, old_bbox, buffer=10):
    x_list = list()
    y_list = list()
    w_list = list()
    h_list = list()
    for bbox in old_bbox:
        if np.sum(bbox) > 0:
            x_list.append(bbox[0])
            y_list.append(bbox[1])
            w_list.append(bbox[0]+bbox[2])
            h_list.append(bbox[1]+bbox[3])

    bbox = list()
    bbox.append(min(x_list))
    bbox.append(min(y_list))
    bbox.append(max(w_list) - bbox[0])
    bbox.append(max(h_list) - bbox[1])
    
    if bbox[0] < buffer:
        bbox[2] += bbox[0] + buffer
        bbox[0] = 0
    else:
        bbox[2] += buffer*2
        bbox[0] -= buffer
            
    if bbox[1] < buffer:
        bbox[3] += bbox[1] + buffer
        bbox[1] = 0
    else:
        bbox[3] += buffer*2
        bbox[1] -= buffer
    
    if bbox[2] + bbox[0] > image.shape[1]:
        bbox[2] = image.shape[1] - bbox[0]
    
    if bbox[3] + bbox[1] > image.shape[0]:
        bbox[3] = image.shape[0] - bbox[1]
    return bbox

def crop_bbox(source_bbox, window, ratio):
    new_bbox = source_bbox
    for b, bbox in enumerate(source_bbox):
        if np.sum(bbox) > 0:
            new_bbox[b, 0] = bbox[0] - window[0]
            new_bbox[b,1] = bbox[1] - window[1]
    return (new_bbox * ratio).astype(np.int8)

def crop_image(image, window, bbox):
    cropped = image[window[1]:window[1]+window[3], window[0]:window[0]+window[2], :]
    resized = np.zeros((64,64,3), dtype=np.float32)
    shape = cropped.shape
    new_box = bbox
#    if (shape[0] > 64) or (shape[1] > 64):
    ratio = 64.0 / max(shape)
    cropped = cv2.resize(cropped, (0, 0), fx=ratio, fy=ratio)
    new_box = crop_bbox(bbox, window, ratio)
    shape = cropped.shape
    resized[0:shape[0], 0:shape[1],:] = cropped
    return resized, new_box

def bbox_patch(subplot, bbox):
    for i in range(bbox.shape[0]):
        if np.sum(bbox[i]) > 0:
            subplot.add_patch(Rectangle((bbox[i][0], bbox[i, 1]), bbox[i][2], 
                                        bbox[i][3], fill=False))

def divide_dataset(dataset, bbox, labels, name, data_dict = {}, parts=4):
    total_len = dataset.shape[0]
    if total_len % parts == 0:
        section_len = total_len // parts
    else:
        section_len = total_len // (parts - 1)
    for p in range(parts):
        start = p * section_len
        end = (p+1) * section_len
        if end > total_len:
            end = total_len
        data_dict[name + '_' + str(p) + '_data'] = dataset[start:end]
        data_dict[name + '_' + str(p) + '_bbox'] = bbox[start:end]
        data_dict[name + '_' + str(p) + '_labels'] = labels[start:end]
    return data_dict

def reassemble_dataset(data_dict):
    key_list = list(data_dict.keys())
    key_dict = {}
    for key in key_list:
        current = key.split(sep='_')
        if int(current[1]) >= key_dict.get(current[0], 0):
            key_dict[current[0]] = int(current[1])
    result = {}
    for key in key_dict:
        data = []
        bbox = []
        label = []
        for count in range(key_dict[key]+1):
            if count == 0:
                data = data_dict[key + '_' + str(count) + '_data']
                bbox = data_dict[key + '_' + str(count) + '_bbox']
                label = data_dict[key + '_' + str(count) + '_labels']
            else:
                data = np.append(data, data_dict[key + '_' + str(count) + '_data'], axis=0)
                bbox = np.append(bbox, data_dict[key + '_' + str(count) + '_bbox'], axis=0)
                label = np.append(label, data_dict[key + '_' + str(count) + '_labels'], axis=0)
        result[key + '_dataset'] = data
        result[key + '_bbox'] = bbox
        result[key + '_labels'] = label
    return result


def prep_dataset(source_dataset, source_bbox, sample=-1):    
    cropped_dataset = list() # np.zeros((data_len, 64, 64), dtype=np.float32)
    cropped_bbox = list() # np.zeros((data_len, 5, 4), dtype=np.int8)
    for filename, filebox in zip(source_dataset, source_bbox):
        try:
            image = load_image(filename[0])
            working_bbox = filebox.copy()
            window = make_window(image, working_bbox)
            cropped, new_bbox = crop_image(image, window, working_bbox)
            cropped_dataset.append(cropped.copy())
            cropped_bbox.append(new_bbox.copy())
        except Exception as e:
            print('There is a problem with: ', filename)
    cropped_dataset = np.asarray(cropped_dataset, dtype=np.float32)
    cropped_bbox = np.asarray(cropped_bbox, dtype=np.int8)
    return cropped_dataset, cropped_bbox
    
def get_dataset(force=False):
    svhn_data = get_svhn.make_composite_dataset('big')
    dataset_name = 'svhn_matrices.npz'
    if force or not os.path.exists(dataset_name):
        train_dataset, train_box = prep_dataset(svhn_data['train_dataset'], svhn_data['train_bbox'], 10)
        valid_dataset, valid_box = prep_dataset(svhn_data['valid_dataset'], svhn_data['valid_bbox'], 10)
        test_dataset, test_box = prep_dataset(svhn_data['test_dataset'], svhn_data['test_bbox'], 10)
        train_labels = svhn_data['train_labels']
        valid_labels = svhn_data['valid_labels']
        test_labels = svhn_data['test_labels']
        dataset = divide_dataset(train_dataset, train_box, train_labels, 'train', parts = 2)
        dataset = divide_dataset(valid_dataset, valid_box, valid_labels, 'valid', data_dict = dataset, parts = 1)
        dataset = divide_dataset(test_dataset, test_box, test_labels, 'test', data_dict = dataset, parts = 1)            

#        dataset = divide_dataset(test_dataset, test_box, test_labels, 'test', parts = 1)            
        try:
            np.savez(dataset_name, **dataset)
        except Exception as e:
            print('Unable to save data to', dataset_name, ':', e)

    try:                 
        dataset = np.load(dataset_name)
    except Exception as e:
      print('Unable to process data from', dataset, ':', e)
      raise
    return reassemble_dataset(dataset)

if __name__ == "__main__":    
    svhn_matrix = get_dataset(force=True)
    for i in svhn_matrix:
        print(i, svhn_matrix[i].shape)
    subplt_h = 2
    subplt_w = 3
    samples = [[(svhn_matrix['train_dataset'][10], svhn_matrix['train_bbox'][10]),
               (svhn_matrix['test_dataset'][10], svhn_matrix['test_bbox'][10]),
               (svhn_matrix['valid_dataset'][10], svhn_matrix['valid_bbox'][10])],
               [(svhn_matrix['train_dataset'][100], svhn_matrix['train_bbox'][100]),
               (svhn_matrix['valid_dataset'][100], svhn_matrix['valid_bbox'][100]),
               (svhn_matrix['test_dataset'][100], svhn_matrix['test_bbox'][100]),]]
    f, axarr = plt.subplots(subplt_h, subplt_w)    
    for r, row in enumerate(samples):
            for c, column in enumerate(row):
                axarr[r, c].imshow(column[0])
                bbox_patch(axarr[r, c], column[1])
