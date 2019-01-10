# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:14:24 2018

@author: shree
"""
import os, json
import numpy as np
import pandas as pd
from collections import defaultdict
import keras
from keras.layers import *
from keras.models import *

rowsSnapshot = 800
colsFeature = 19

def feature_generation(bt_data, lt_data, c_data, d_data, data):

    bt_lt_dist = np.vstack([np.sqrt(np.sum((p1[0:2]-p2[0:2])**2)) for p1, p2 in zip(bt_data, lt_data)])
    c_lt_dist = np.vstack([np.sqrt(np.sum((p1[0:2]-p2[0:2])**2)) for p1, p2 in zip(c_data, lt_data)])
    c_bt_dist = np.vstack([np.sqrt(np.sum((p1[0:2]-p2[0:2])**2)) for p1, p2 in zip(c_data, bt_data)])
    
    bt_velocity = np.vstack([np.sqrt(np.sum((p1[0:2]-p2[0:2])**2)) for p1, p2 in zip(bt_data[:-1], bt_data[1:])])
    lt_velocity = np.vstack([np.sqrt(np.sum((p1[0:2]-p2[0:2])**2)) for p1, p2 in zip(lt_data[:-1], lt_data[1:])])
    c_velocity = np.vstack([np.sqrt(np.sum((p1[0:2]-p2[0:2])**2)) for p1, p2 in zip(c_data[:-1], c_data[1:])])

    bt_velocity = np.append([[0]], bt_velocity, axis=0)
    lt_velocity = np.append([[0]], lt_velocity, axis=0)
    c_velocity = np.append([[0]], c_velocity, axis=0)
    
    bt_acce = np.vstack([p1 - p2 for p1, p2 in zip(bt_velocity[:-1], bt_velocity[1:])])
    lt_acce = np.vstack([p1 - p2 for p1, p2 in zip(lt_velocity[:-1], lt_velocity[1:])])
    c_acce = np.vstack([p1 - p2 for p1, p2 in zip(c_velocity[:-1], c_velocity[1:])])

    bt_acce = np.append([[0]], bt_acce, axis=0)
    lt_acce = np.append([[0]], lt_acce, axis=0)
    c_acce = np.append([[0]], c_acce, axis=0)
    data = np.concatenate([data, bt_lt_dist, c_lt_dist, c_bt_dist, bt_velocity, lt_velocity, 
                           c_velocity, bt_acce, lt_acce, c_acce], axis = -1)
    
    return data

def parse_animations(animation_files, dictLabel_to_Idx,
                     shape_data_columns=['big_triangle_XYR','little_triangle_XYR','circle_XYR','door_XYR']):
    dictData = defaultdict(dict)
    countData = 0
    for animation_file in animation_files:
        with open(animation_file, 'r') as f:
            animation = json.load(f)

        labels = [frame[0] for frame in animation] # first column is sequence of action labels
        data = np.array([frame[1:11] for frame in animation]) # columns 2,3,4 are X, Y, R values for big triangle        
        
        bt_data = np.array([frame[1:4] for frame in animation]) # columns 2,3,4 are X, Y, R values for big triangle
        lt_data = np.array([frame[4:7] for frame in animation]) # columns 5,6,7 are X, Y, R values for little triangle
        c_data = np.array([frame[7:10] for frame in animation]) # columns 8,9,10 are X, Y, R values for circle
        d_data = np.array([frame[10] for frame in animation])[:, None] # column 11 is R value for door
        #pad door data with 0's so data can ultimately be processed in matrix format
        d_data = np.concatenate([np.zeros((len(d_data), 2)), d_data], axis=-1)
        
        data = feature_generation(bt_data, lt_data, c_data, d_data, data)
        
        padZeroes = np.array([0] * 19)
        snapshots = []
        for i in range(len(labels)-1):
            if labels[i] != labels[i+1]:
                snapshots.append(data[i])
                
                while len(snapshots) <= rowsSnapshot:
                    snapshots.append(padZeroes)
                if len(snapshots) > rowsSnapshot:
                    snapshots = snapshots[-rowsSnapshot:]
                
                if labels[i] not in dictLabel_to_Idx:    
                    dictData[countData]['label'] = dictLabel_to_Idx['Unknown']
                else:
                    dictData[countData]['label'] = dictLabel_to_Idx[labels[i]]
                dictData[countData]['data'] = snapshots
                countData +=1
                snapshots = []
            else:
                snapshots.append(data[i])
                   
    animation_data = pd.DataFrame.from_dict(dictData,orient='index')
    
    return animation_data

def getLabels(animation_files):
    dictLabel_to_Idx = {}
    dictLabel_to_Idx['Unknown'] = 0
    countLabel = 1
    for animation_file in animation_files:
        with open(animation_file, 'r') as f:
            animation = json.load(f)
            
        labels = [frame[0] for frame in animation] # first column is sequence of action labels
        for label in labels:
            if label not in dictLabel_to_Idx:
                dictLabel_to_Idx[label] = countLabel
                countLabel += 1
    dictIdx_to_Label = {v:k for k,v in dictLabel_to_Idx.items()}
    return dictLabel_to_Idx, dictIdx_to_Label

"""Load the training and testing data, put in matrix format"""
animation_filenames = ['example_animations/' + filename for filename in os.listdir('example_animations')]
n_test_animations = 1
labels_to_idxs, idxs_to_labels = getLabels(animation_filenames[:-n_test_animations])

"""Segregate test and training data"""
raw_train_data = parse_animations(animation_filenames[:-n_test_animations], labels_to_idxs)
raw_test_data = parse_animations(animation_filenames[-n_test_animations:], labels_to_idxs)

"""Divide data between features(x) and corresponding labels(y)"""
y_train = np.vstack(raw_train_data['label'].values)
x_train = np.stack(raw_train_data['data'].values)

y_test = np.vstack(raw_test_data['label'].values)
x_test = np.stack(raw_test_data['data'].values)

"""Incorporate channel as last"""
x_train = x_train.reshape(x_train.shape[0], rowsSnapshot, colsFeature, 1)
x_test = x_test.reshape(x_test.shape[0], rowsSnapshot, colsFeature, 1)
input_shape = (rowsSnapshot, colsFeature, 1)
    
print('X_train shape:', x_train.shape)
print('X_test shape:', x_test.shape)

"""convert class vectors to binary class matrices"""
num_category = len(labels_to_idxs)
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

print('y_train shape:', y_train.shape)
print('y_test_shape:', y_test.shape)

"""2D CNN Model Building"""
model = Sequential()
#convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#32 convolution filters used each of size 3x3
#again
model.add(Conv2D(64, (3, 3), activation='relu'))
#64 convolution filters used each of size 3x3
#choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))
#flatten since too many dimensions, we only want a classification output
model.add(Flatten())
#fully connected to get all relevant data
model.add(Dense(800, activation='relu'))
#one more dropout for convergence' sake :) 
model.add(Dropout(0.5))
#output a softmax to squash the matrix into output probabilities
model.add(Dense(num_category, activation='softmax'))
 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print(model.summary())

"""Model Training"""
batch_size = 10
num_epoch = 10
model_log = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          validation_data=(x_test, y_test))

"""Model Evaluation"""
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0]) 
print('Test accuracy:', score[1]) 

#"""Model Prediction"""
#n_best=1
#all_best_labels = []
#all_best_probs = []
#pred_probs = model.predict(x_test)
#for probs in pred_probs:
#    best_labels = np.argsort(probs)[::-1][:n_best]
#    all_best_labels.append(best_labels)
#    best_probs = probs[best_labels]
#    all_best_probs.append(best_probs)
#all_best_labels = np.array(all_best_labels)
#all_best_probs = np.array(all_best_probs)
#
