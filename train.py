#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:30:10 2018

@author: chris
"""

import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from models import ResNet
from keras.utils import to_categorical, plot_model
from time import time

filters_list = [4,8,16]
input_size = (50,50,1)
output_size = 12

sr = ResNet(filters_list, input_size, output_size)
sr.m.compile(loss='categorical_crossentropy', 
             optimizer='adadelta', 
             metrics=["accuracy"])

date = '1301'
arch = 'resnet'

plot_model(sr.m, to_file = './models/{}_{}.png'.format(arch, date), 
           show_shapes = True) 

#callbacks 
checkpointer = ModelCheckpoint(filepath="./models/{}_{}.h5".format(arch, date),
                               verbose=0,
                               save_best_only=True)
   
earlystopping = EarlyStopping()

tensorboard = TensorBoard(log_dir = "./logs/{}_{}".format(date, time()), 
                          histogram_freq = 0, 
                          write_graph = True, 
                          write_images = True)


history = sr.m.fit(X_train, 
                   to_categorical(Y_train), 
                   batch_size = batch_size, 
                   epochs = 25, 
                   verbose = 1, shuffle = True, 
                   class_weight = class_weights,
                   validation_data = (X_valid, to_categorical(Y_valid)), 
                   callbacks = [checkpointer])