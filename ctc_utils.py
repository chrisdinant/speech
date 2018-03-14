#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 09:10:28 2018

@author: chris
"""
import numpy as np

#%%
# From Baidu ba-dls-deepspeech - https://github.com/baidu-research/ba-dls-deepspeech
# Character map list

char_map_str = """
<SPACE> 0
a 1
b 2
c 3
d 4
e 5
f 6
g 7
h 8
i 9
j 10
k 11
l 12
m 13
n 14
o 15
p 16
q 17
r 18
s 19
t 20
u 21
v 22
w 23
x 24
y 25
z 26
' 27

"""

char_map = {}
index_map = {}

for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)] = ch

index_map[0] = ' '

#%%
def text_to_int(text):
    """
    takes the character map and returns a series of 
    integers for the inserted text
    the 'silence' class returns only 27's
    """
    int_seq = []
    if text == 'silence':
        for r in range(8):
            int_seq.append(27)
    else:
        for c in text:
            ch = char_map[c]
            int_seq.append(ch)
    return int_seq
        
def get_intseq(trans, max_len = 8):
    """
    pads integer list with 27's up to max length
    """
    t = text_to_int(trans)
    while (len(t) < max_len):
        t.append(27)
    return t

def get_ctc_params(Y, classes_list, len_char_map = 28):
    """
    Usage:
        creates parameters required for K.ctc_batch_cost function 
    Args:
        Y (ndarray): target set with all classes
        classes_list (list): list with class names
        len_char_map (int): length of the character map
    Returns:
        3 ndarrays
    """
    labels = np.array([get_intseq(classes_list[Y[l]]) for l, _ in enumerate(Y)])
    input_length = np.array([len_char_map for _ in Y])
    label_length = np.array([8 for _ in Y])
    return labels, input_length, label_length



























