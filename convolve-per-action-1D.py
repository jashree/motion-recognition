# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:14:24 2018

@author: shree
"""
import os, json
import numpy as np
import pandas as pd
from collections import defaultdict
from keras.layers import Input, Flatten, Dense, Reshape, Conv1D, MaxPooling1D
from keras.models import Model

rowsSnapshot = 800

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

def create_cnn_model(len_snapshot, len_features, n_labels, filters=5, 
                     kernel_size=10, strides=1, pool_size=1, n_hidden_layers=2, n_hidden_dim=1000):
    input_layer = Input(shape=(len_snapshot, len_features), name='input')
    reshape_layer = Reshape((len_snapshot, -1))(input_layer)
    conv_layer = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, 
                        activation='sigmoid', padding='same', name='convolution')(reshape_layer)
    pool_layer = MaxPooling1D(pool_size=pool_size, padding='same', name='pool')(conv_layer)
    flatten_layer = Flatten(name='flatten')(pool_layer)
    hidden_layer = Dense(units=n_hidden_dim, activation='sigmoid', name='hidden1')(flatten_layer)
    for layer_idx in range(1, n_hidden_layers):
        hidden_layer = Dense(units=n_hidden_dim, activation='sigmoid', 
                             name='hidden' + str(layer_idx + 1))(hidden_layer)
    output_layer = Dense(units=n_labels, activation='softmax', name='output')(hidden_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')
    print(model.summary())
    return model

def evaluate_prediction(model, x, labels):
    pred_probs = model.predict(x)
    pred_probs = pred_probs.reshape((-1, pred_probs.shape[-1]))
    labels = labels.flatten().astype(int)
    accuracy = np.mean(np.argmax(pred_probs, axis=-1) == labels)
    pred_probs = pred_probs[np.arange(len(labels)), labels]
    perplexity = np.exp(-np.mean(np.log(pred_probs)))
    return accuracy, perplexity

def predict_labels(model, animation, n_best=1, recurrent=False):
    all_best_labels = []
    all_best_probs = []
    if recurrent: #if model is RNN, append new dimension
        pred_probs = model.predict(animation[None])[0]
    else:
        pred_probs = model.predict(animation)
    for probs in pred_probs:
        best_labels = np.argsort(probs)[::-1][:n_best]
        all_best_labels.append(best_labels)
        best_probs = probs[best_labels]
        all_best_probs.append(best_probs)
    all_best_labels = np.array(all_best_labels)
    all_best_probs = np.array(all_best_probs)
    return all_best_labels, all_best_probs

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
print(x_train.shape, y_train.shape)

y_test = np.vstack(raw_test_data['label'].values)
x_test = np.stack(raw_test_data['data'].values)
print(x_test.shape, y_test.shape)

model = create_cnn_model(len_snapshot=rowsSnapshot,
                         len_features=19,
                         n_labels=len(labels_to_idxs))

"""Train CNN model"""
loss = model.fit(x=x_train, y=y_train,
                 epochs=5, batch_size=100, verbose=0)
print(loss.history)

"""Divide test data into snapshots as with training animations"""
accuracy, perplexity = evaluate_prediction(model, 
                                           x_test,y_test)
print(accuracy, perplexity)

test_animation = x_test
test_animation_labels = y_test
pred_labels, pred_probs = predict_labels(model, test_animation, n_best=1)

for each in list(zip([[idxs_to_labels[label_idx] for label_idx in labels] for labels in pred_labels],
         pred_probs,
         [[idxs_to_labels[label_idx] for label_idx in labels] for labels in test_animation_labels])):
    print(each)
