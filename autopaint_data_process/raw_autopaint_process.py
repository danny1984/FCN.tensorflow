# -*- coding: UTF-8 -*-
#!/bin/python

import cv2
import matplotlib.pyplot as plt
RAW_PAINT_DATA_PATH="./data/raw_paint_data/"
RAW_PAINT_DATA_OUTPUT_PATH="./data/raw_train_data/"


# step 2. Random Crop
INPUT_SIZE = 512
RAW_RC_DATA_INPUT_PATH = RAW_PAINT_DATA_OUTPUT_PATH
RAW_RC_DATA_OUTPUT_PATH = "./data/rc_train_data/"
# 图像 0
raw_pre = "raw_1_"
raw_train = RAW_RC_DATA_INPUT_PATH + raw_pre + "train.jpg"
raw_label = RAW_RC_DATA_INPUT_PATH + raw_pre + "label.png"

img_train = cv2.imread(raw_train)
img_label = cv2.imread(raw_label, cv2.IMREAD_GRAYSCALE)
crop_w = INPUT_SIZE
crop_h = INPUT_SIZE
img_h, img_w = img_train.shape[:2]
img_h_range = img_h - INPUT_SIZE
img_w_range = img_w - INPUT_SIZE
stride_size = 50

## Create training data
train_img_file_list = RAW_RC_DATA_OUTPUT_PATH + "train_img_file_list.txt"
label_img_file_list = RAW_RC_DATA_OUTPUT_PATH + "label_img_file_list.txt"
fin_train_file = open(train_img_file_list, "a+")
fin_label_file = open(label_img_file_list, "a+")
cnt = 0
for x in xrange(0, img_h_range, stride_size):
    for y in xrange(0, img_w_range, stride_size):
        cropped_train = img_train[x:(x + INPUT_SIZE), y:(y + INPUT_SIZE)]
        cropped_label = img_label[x:(x + INPUT_SIZE), y:(y + INPUT_SIZE)]
        cnt = cnt + 1
        train_file_name = RAW_RC_DATA_OUTPUT_PATH + "/images/" + raw_pre + "train_" + str(cnt) + ".png"
        label_file_name = RAW_RC_DATA_OUTPUT_PATH + "/labels/" + raw_pre + "label_" + str(cnt) + ".png"
        fin_train_file.write(train_file_name + '\n')
        fin_label_file.write(label_file_name + '\n')
        cv2.imwrite(train_file_name, cropped_train)
        cv2.imwrite(label_file_name, cropped_label)

fin_train_file.close()
fin_label_file.close()

print "Generate " + str(cnt) + " <train, label> pairs"

