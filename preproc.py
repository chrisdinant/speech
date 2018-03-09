#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:24:54 2018

@author: christoffel
"""

import numpy as np
import librosa
import librosa.display
import os
import csv
from collections import Counter

train_dir = './data/train/audio/' #download files from kaggle
folders = os.listdir(train_dir)
classes = ['yes', 'no', 
           'up', 'down', 
           'left', 'right', 
           'on', 'off', 
           'stop', 'go', 
           'silence', 'unknown']


#%%
def make_spec(file, file_dir=train_dir, flip = False, ps = False, st = 4):
    """
    create a melspectrogram from the amplitude of the sound
    
    Args:
        file (str): filename
        file_dir (str): directory path
        flip (bool): reverse time axis
        ps (bool): pitch shift
        st (int): half-note steps for pitch shift
    Returns:
        np.array with shape (122,85) (time, freq)
    """
    sig, rate = librosa.load(file_dir + file, sr = 16000)
    if len(sig) < 16000: # pad shorter than 1 sec audio with zeros
        sig = np.pad(sig, (0,16000-len(sig)), 'constant', constant_values = 0)
    if ps:
        sig = librosa.effects.pitch_shift(sig, rate, st)
    D = librosa.amplitude_to_db(librosa.stft(sig[:16000], n_fft=512, 
                                             hop_length = 128, 
                                             center = False), ref = np.max)
    S = librosa.feature.melspectrogram(S=D, n_mels = 85).T
    if flip:
        S = np.flipud(S)
    return S.astype(np.float32)
    


#%%
def create_silence():
    
#%% list all files


    
#%%
def create_sets():
        