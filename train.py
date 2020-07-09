# -*- coding: utf-8 -*-
"""
Created on Mon Jul 6 13:55:05 2020

@author: xiaobo
@email: linxiaobo110@gmail.com
"""

# Copyright (C)
#
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import random
import time
import matplotlib.pyplot as plt

# the image size for netwokr-input
img_size_net = 128
# the train batch
batch_size = 32
# the path of dataset
dataset_path = './dataset/'
# path of each kind in the dataset_path
sorts_list = ['paper', 'rock', 'scissors']
# name for lables, maybe Chinese
wordlist = ['paper', 'rock', 'scissors']
# path for result
run_path = './run/'
if not os.path.exists(run_path):
    os.mkdir(run_path)
    
####################################################
# prepare the train data

# 1. read data_set
def load_valid_data(data_path):
    # num of each kind of samples
    cnt_each = np.zeros(3, dtype=np.int)
    # num of all samples
    img_cnt = 0
    # counter for lable
    label_cnt = 0
    test_images = []
    test_lables = []
    for sort_path in sorts_list:    
        flower_list = os.listdir(data_path + sort_path)
        for img_name in flower_list:
            img_path = data_path + sort_path + "/" + img_name
            img = cv2.imread(img_path)  
            img_scale = cv2.resize(img,(img_size_net, img_size_net), interpolation = cv2.INTER_CUBIC)
            if not img is None:
                test_images.append(img_scale)
                test_lables.append(label_cnt)
                
                # static the num of different lable
                cnt_each[label_cnt] += 1
                
                # print one image every 100 examples
                if img_cnt % 100 == 0:
                    print('The ', str(img_cnt), ' image')
                    plt.figure('x')
                    plt.imshow(cv2.cvtColor(img_scale, cv2.COLOR_BGR2RGB))
                    plt.show()
                img_cnt += 1
                
        label_cnt += 1     
    print('The samples in the data contain: paper-', cnt_each[0], ', rock-', cnt_each[1], ', scissors-', cnt_each[2])
    return test_images, test_lables

(validSet_images, validSet_lables) = load_valid_data(dataset_path)
dataSet_img = np.array(validSet_images)
dataSet_lable = np.array(validSet_lables)

from sklearn.utils import shuffle
dataSet_img,dataSet_lable = shuffle(dataSet_img,dataSet_lable)
dataSet_img = np.array(dataSet_img, dtype=np.float32)
dataSet_lable = np.array(dataSet_lable)
dataSet_img = dataSet_img / 255.

# 2. devide the train_set and test_set
dataset_nums = len(dataSet_lable)
trainSet_num = int(0.75 * dataset_nums)
trainSet_img = dataSet_img[0 : trainSet_num, :, :, :]
testSet_img = dataSet_img[trainSet_num : , :, :, :]
trainSet_label = dataSet_lable[0 : trainSet_num]
testSet_label = dataSet_lable[trainSet_num : ]

# 3. show the distrute of different kinds of example
l = []
for x in dataSet_lable:
    l.append(wordlist[x])
plt.hist(l, rwidth=0.5)
plt.show()


#########################################################
# train modle
model = keras.Sequential([
    #keras.layers.Flatten(input_shape=(128, 128, 3)),
    keras.layers.Conv2D(32, (3,3), padding="same", input_shape=(img_size_net, img_size_net, 3), name='x_input', activation=tf.nn.relu),
#     keras.layers.BatchNormalization(),
#     keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, (3,3), padding="same", activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(128, (3,3), padding="same", activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(128, (3,3), padding="same", activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(200, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    # 最后一个层决定输出类别的数量
    keras.layers.Dense(3, activation=tf.nn.softmax, name='y_out')
])
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.summary()

print('First train: ')
history = model.fit(trainSet_img, trainSet_label, 
                    batch_size=batch_size, 
                    epochs=30, 
                    validation_data=(testSet_img, testSet_label)
                    )
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'][1:])
plt.plot(history.history['val_accuracy'][1:])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()

#######################################################################
# save trained model

model_path = run_path + "model.h5"
model.save(model_path)
print('The trained result is saved on ', os.path.join(os.getcwd(), model_path))