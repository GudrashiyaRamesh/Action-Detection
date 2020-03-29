# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:28:14 2020

@author: RAMESH
"""
import pandas as pd
import numpy as np
import cv2
import os
import h5py
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam
from keras.utils.io_utils import HDF5Matrix

SEQ_LEN = 30
MAX_SEQ_LEN = 200
BATCH_SIZE = 16
EPOCHS = 1000

def preprocess_image(img1):
    img = cv2.resize(img1, (299,299))
    return img

def get_label_index():
    class_name = ["Criminal" , "Safe"]  
    index = np.arange(0, len(class_name))
    label_index = dict(zip(class_name, index))
    return label_index

def encode_video(,model,label_index):
    
    video = cv2.VideoCapture("130.mp4")
    c = 7
    images = []
    count = 0
    while(video.isOpened()):
        ret , frame = video.read()
        if ret == True:
            if c == 0:
                frame = preprocess_image(frame)
                images.append(frame)
                count +=1
                c = 7
                if count == 30:
                    break
            c -= 1
        else:
            break
                
    video.release()
    cv2.destroyAllWindows()
    
    features = model.predict(np.array(images))
    index = label_index[row["class"].iloc[0]]
    print(index)
    
    return features, index

from keras.utils import np_utils
def encode_dataset(data, model, label_index, phase):
    input_f = []
    output_y = []
    required_classes = ["Criminal" , "Safe"]
   
    
    for i in tqdm(range(data.shape[0])):
    # Check whether the given row , is of a class that is required
        if str(data.iloc[[i]]["class"].iloc[0]) in required_classes:
 
            features,y =  encode_video(data.iloc[[i]], model, label_index)
            input_f.append(features)
            output_y.append(y)
        
    
    le_labels = np_utils.to_categorical(output_y)
    f = h5py.File(phase+'_8'+'.h5', 'w')
    f.create_dataset(phase, data=np.array(input_f))
    f.create_dataset(phase+"_labels", data=np.array(le_labels))
    
    del input_f[:]
    del output_y[:]

def main():
     # Get model with pretrained weights.
    base_model = InceptionV3(
    weights='imagenet',
    include_top=True)
    
    # We'll extract features at the final pool layer.
    model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('avg_pool').output)
    
    label_index = get_label_index()
    
    encode_dataset(model,label_index,"train")
    encode_dataset(model,label_index,"test")