# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:45:20 2020

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


def get_data(path, if_pd=False):
    """Load our data from file."""
    names = ['partition', 'class', 'video_name', 'frames']
    df = pd.read_csv(path,names=names)
    return df

def get_class_dict(df):
    class_name =  list(df['class'].unique())
    index = np.arange(0, len(class_name))
    label_index = dict(zip(class_name, index))
    index_label = dict(zip(index, class_name))
    return (label_index, index_label)
    
def clean_data(df):
    mask = np.logical_and(df['frames'] >= SEQ_LEN, df['frames'] <= MAX_SEQ_LEN)
    df = df[mask]
    return df
def split_train_test(df):
    partition =  (df.groupby(['partition']))
    un = df['partition'].unique()
    train = partition.get_group(un[0])
    test = partition.get_group(un[1])
    return (train, test)

def preprocess_image(img1):
    img = cv2.resize(img1, (299,299))
    return img

def encode_video(row, model, label_index):
    cap = cv2.VideoCapture(os.path.join("Human-Action-Classification--master/data","UCF-101",str(row["class"].iloc[0]) ,str(row["video_name"].iloc[0]) + ".avi"))
    images = []  
    for i in range(SEQ_LEN):
        ret, frame = cap.read()
        frame = preprocess_image(frame)
        images.append(frame)
    
    
    features = model.predict(np.array(images))
    index = label_index[row["class"].iloc[0]]
    print(index)
    #y_onehot = to_categorical(index, len(label_index.keys()))
    
    return features, index

from keras.utils import np_utils
def encode_dataset(data, model, label_index, phase):
    input_f = []
    output_y = []
    required_classes = ["ApplyEyeMakeup" , "ApplyLipstick" , "Archery" , "BabyCrawling" , "BalanceBeam" ,
                       "BandMarching" , "BaseballPitch" , "Basketball" , "BasketballDunk"]
   
    
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
    
    # Getting the data
    df = get_data('Human-Action-Classification--master/data/data_file.csv')
    
    # Clean the data
    df_clean = clean_data(df)
    
    # Creating index-label maps and inverse_maps
    label_index, index_label = get_class_dict(df_clean)
    
    # Split the dataset into train and test
    train, test = split_train_test(df_clean)
    
    # Encoding the dataset
    encode_dataset(train, model, label_index, "train")
    encode_dataset(test,model,label_index,"test")

main()



x_train = HDF5Matrix('train_8.h5', 'train')
y_train = HDF5Matrix('train_8.h5', 'train_labels')
x_test = HDF5Matrix('test_8.h5', 'test')
y_test = HDF5Matrix('test_8.h5', 'test_labels')


from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.layers import LSTM


def lstm():
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predominantly."""
    input_shape = (SEQ_LEN, 2048)
    # Model.
    model = Sequential()
    model.add(LSTM(2048,input_shape=input_shape,dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))
    #model.add(Dense(10, activation='softmax'))"""
    
    optimizer = Adam(lr=1e-5, decay=1e-6)
    metrics = ['accuracy', 'top_k_categorical_accuracy']
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=metrics)
    return model

model = lstm()
#model.fit(x_train, y_train)
model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = 100,verbose = 2,validation_data = (x_test, y_test),shuffle = 'batch')
        
model.save("Activity_Recognition.h5")



model1 = load_model("Activity_Recognition.h5")

def prepare_video(filepath):
    cap = cv2.VideoCapture("bas.avi")
    images = []  
    count = 0
    farray = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True: 
            frame = preprocess_image(frame)
            images.append(frame)
            count += 1
            
            if count >= 30: 
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    features = model.predict(np.array(images))
    farray.append(features.tolist())
    
    arr = np.asarray(farray)
    return arr
    

prediction = model1.predict(prepare_video("bas.avi"))

print(prediction.max())
    