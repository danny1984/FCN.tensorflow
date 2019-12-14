__author__ = 'Danny'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

def read_autopaint_dataset(data_dir):
    data_pre = "./data/rc_train_data/"
    autopaint_train_file = os.path.join(data_dir, data_pre, "train_img_file_list.txt")
    autopaint_label_file = os.path.join(data_dir, data_pre, "label_img_file_list.txt")

    with open(autopaint_train_file, 'r') as fin_train:
        image_file_list = fin_train.readlines()
        image_file_list = np.array([file[:-1] for file in image_file_list])
    
    with open(autopaint_label_file, 'r') as fin_label:
        label_file_list = np.array(fin_label.readlines())
        label_file_list = np.array([file[:-1] for file in label_file_list])
        
    num = len(image_file_list)
    
    # shuffle 
    perm = np.arange(num)
    np.random.shuffle(perm)
    image_file_list = image_file_list[perm]
    label_file_list = label_file_list[perm]
    
    # train&validation splitting
    train_offset = num / 10 * 9
    train_img_list = image_file_list[:train_offset]
    valid_img_list = image_file_list[train_offset:]
    train_lbl_list = label_file_list[:train_offset]
    valid_lbl_list = label_file_list[train_offset:]

    return train_img_list, train_lbl_list, valid_img_list, valid_lbl_list

# deprecated
def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list
