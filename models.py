#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:43:51 2018

@author: chris
"""

from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model


#%%
class ResNet():
    """
    Usage: 
        sr = ResNet([4,8,16], input_size=(50,50,1), output_size=12)
        followed by sr.m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
        save plotted model with: keras.utils.plot_model(sr.m, to_file = '<location>.png', show_shapes=True)
    """
    def __init__(self,
                 filters_list=[], 
                 input_size=None, 
                 output_size=None,
                 initializer='glorot_uniform'):
        self.filters_list = filters_list
        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.m = None        
    
    def _block(self, filters, inp):
        """ one residual block in a ResNet
        
        Args:
            filters (int): number of convolutional filters
            inp (tf.tensor): output from previous layer
            
        Returns:
            tf.tensor: output of residual block
        """
        layer_1 = BatchNormalization()(inp)
        act_1 = Activation('relu')(layer_1)
        conv_1 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_1)
        layer_2 = BatchNormalization()(conv_1)
        act_2 = Activation('relu')(layer_2)
        conv_2 = Conv2D(filters, (3,3), 
                        padding = 'same', 
                        kernel_initializer = self.initializer)(act_2)
        return(conv_2)

    def build(self):
        """
        Returns:
            keras.engine.training.Model
        """
        i = Input(shape=self.input_size, name = 'input')
        x = Conv2D(self.filters_list[0], (3,3), 
                   padding = 'same', 
                   kernel_initializer = self.initializer)(i)
        x = MaxPooling2D(padding = 'same')(x)        
        x = Add()([self._block(self.filters_list[0], x),x])
        x = Add()([self._block(self.filters_list[0], x),x])
        x = Add()([self._block(self.filters_list[0], x),x])
        for filt in self.filters_list[1:]:
            x = Conv2D(filt, (3,3),
                       strides = (2,2),
                       padding = 'same',
                       activation = 'relu',
                       kernel_initializer = self.initializer)(x)
            x = Add()([self._block(filt, x),x])
            x = Add()([self._block(filt, x),x])
            x = Add()([self._block(filt, x),x])
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.output_size, activation = 'softmax')(x)
        
        self.m = Model(i,x)
        return self.m
           
    
#%%
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        