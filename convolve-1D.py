#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 07 14:27:27 2018
@author: jshree
"""
import json, numpy, pandas, os
from keras.layers import *
from keras.models import Model
import MeanDelay

def parse_animations(animation_files, 
                     shape_data_columns=['big_triangle_X','big_triangle_Y', 'big_triangle_R',
                                         'little_triangle_X','little_triangle_Y', 'little_traiagle_R',
                                         'circle_X', 'circle_Y', 'circle_R', 'door_R',
                                         'bt_lt_ditance', 'bt_c_distance', 'lt_c_distance']):
    animation_data = []
    for animation_file in animation_files:
        with open(animation_file, 'r') as f:
            animation = json.load(f)

        labels = [frame[0] for frame in animation]                 
        bt_data_X = [frame[1] for frame in animation]
        bt_data_Y = [frame[2] for frame in animation]
        bt_data_R = [frame[3] for frame in animation]
        lt_data_X = [frame[4] for frame in animation]
        lt_data_Y = [frame[5] for frame in animation]
        lt_data_R = [frame[6] for frame in animation]
        c_data_X = [frame[7] for frame in animation]
        c_data_Y = [frame[8] for frame in animation]
        c_data_R = [frame[9] for frame in animation]
        d_data_R = [frame[10] for frame in animation]
        bt_lt_dist = [numpy.sqrt(numpy.sum((a-b)**2)) for a,b in
                      zip(numpy.array(list(zip(numpy.array(bt_data_X),numpy.array(bt_data_Y)))),
                                   numpy.array(list(zip(numpy.array(lt_data_X),numpy.array(lt_data_Y)))))]
        bt_c_dist = [numpy.sqrt(numpy.sum((a-b)**2)) for a,b in
                      zip(numpy.array(list(zip(numpy.array(bt_data_X),numpy.array(bt_data_Y)))),
                                   numpy.array(list(zip(numpy.array(c_data_X),numpy.array(c_data_Y)))))]
        lt_c_dist = [numpy.sqrt(numpy.sum((a-b)**2)) for a,b in
                      zip(numpy.array(list(zip(numpy.array(lt_data_X),numpy.array(lt_data_Y)))),
                                   numpy.array(list(zip(numpy.array(c_data_X),numpy.array(c_data_Y)))))]

        animation_data.append([labels, bt_data_X, bt_data_Y, bt_data_R, lt_data_X, lt_data_Y, lt_data_R,
                               c_data_X, c_data_Y, c_data_R, d_data_R, bt_lt_dist, bt_c_dist, lt_c_dist])

    animation_data = pandas.DataFrame(animation_data, 
                                      columns=['labels'] + shape_data_columns)
    return animation_data


def get_label_idx_alignment(labels):
    # Reserve index 0 for labels that are not in the training data 
    # This is necessary because animations with these labels could show up in the test set
    labels_to_idxs = {'<UNKNOWN>': 0}
    cur_label_idx = 1
    for animation in labels:
        for label in animation:
            if label not in labels_to_idxs:
                labels_to_idxs[label] = cur_label_idx
                cur_label_idx += 1
    
    idxs_to_labels = {idx:label for label, idx in labels_to_idxs.items()}
    return labels_to_idxs, idxs_to_labels


def get_x_labels(raw_data, labels_to_idxs):
    x = numpy.stack([numpy.stack(animation, axis=-1) for animation in raw_data.iloc[:,1:].values])
    labels = transform_labels_to_idxs(raw_data.iloc[:,0], labels_to_idxs)
    return x, labels

def transform_labels_to_idxs(labels, labels_to_idxs):    
    labels = numpy.stack(labels.apply(lambda animation_labels: numpy.array([labels_to_idxs[label] 
                                      if label in labels_to_idxs else 0
                                      for label in animation_labels])[:,None]).as_matrix())
    return labels 

def get_animation_snapshots(x, labels, n_snapshot_frames=100):
    all_snapshots = []
    all_snapshot_labels = []
    for animation, animation_labels in zip(x, labels):
        assert(len(animation.shape) <= 3)
        snapshots = []
        snapshot_labels = []
        for frame_idx in range(len(animation) - n_snapshot_frames):
            snapshot = animation[frame_idx:frame_idx + n_snapshot_frames]
            snapshots.append(snapshot)
            snapshot_labels.append(animation_labels[frame_idx])
        assert(len(snapshots) == len(snapshot_labels))
        all_snapshots.append(numpy.array(snapshots))
        all_snapshot_labels.append(numpy.array(snapshot_labels))
    all_snapshots = numpy.array(all_snapshots)
    all_snapshot_labels = numpy.array(all_snapshot_labels)
    assert(len(all_snapshots) == len(all_snapshot_labels))
    return all_snapshots, all_snapshot_labels


def create_cnn_model(n_snapshot_frames, n_frame_features, n_labels):
    filters=5
    kernel_size=(10,n_frame_features)
    strides=(1,n_frame_features)
    pool_size=(5,n_frame_features)
    n_hidden_layers=1
    n_hidden_dim=300

    input_layer = Input(shape=(n_snapshot_frames, n_frame_features, 1), name='input')
    reshape_layer = Reshape((n_snapshot_frames, n_frame_features, 1))(input_layer)
    conv_layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                        activation='sigmoid', padding='same', name='convolution')(reshape_layer)
    pool_layer = MaxPool2D(pool_size=pool_size, padding='same', name='pool')(conv_layer)
    flatten_layer = Flatten(name='flatten')(pool_layer)
    hidden_layer = Dense(units=n_hidden_dim, activation='sigmoid', name='hidden1')(flatten_layer)
    for layer_idx in range(1, n_hidden_layers):
        hidden_layer = Dense(units=n_hidden_dim, activation='sigmoid', 
                             name='hidden' + str(layer_idx + 1))(hidden_layer)
    output_layer = Dense(units=n_labels, activation='softmax', name='output')(flatten_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')
    print(model.summary())
    return model

def evaluate_prediction(model, x, labels):
    pred_probs = model.predict(x)
    pred_probs = pred_probs.reshape((-1, pred_probs.shape[-1]))
    labels = labels.flatten().astype(int)
    accuracy = numpy.mean(numpy.argmax(pred_probs, axis=-1) == labels)
    pred_probs = pred_probs[numpy.arange(len(labels)), labels]
    perplexity = numpy.exp(-numpy.mean(numpy.log(pred_probs)))
    return accuracy, perplexity

def predict_labels(model, animation, n_best=1, recurrent=False):
    all_best_labels = []
    all_best_probs = []
    pred_probs = model.predict(animation)
    for probs in pred_probs:
        best_labels = numpy.argsort(probs)[::-1][:n_best]
        all_best_labels.append(best_labels)
        best_probs = probs[best_labels]
        all_best_probs.append(best_probs)
    all_best_labels = numpy.array(all_best_labels)
    all_best_probs = numpy.array(all_best_probs)
    return all_best_labels, all_best_probs


"""Load the training and testing data, put in matrix format"""
animation_filenames = ['example_animations/' + filename for filename in os.listdir('example_animations')]
n_test_animations = 1
raw_train_data = parse_animations(animation_filenames[:-n_test_animations])
raw_test_data = parse_animations(animation_filenames[-n_test_animations:])

"""Calculate mean delay """
mean_delay = MeanDelay.find_delay(animation_filenames)
print("Mean delay is :", mean_delay)

"""Translate between string action labels and numerical indices"""
labels_to_idxs, idxs_to_labels = get_label_idx_alignment(raw_train_data['labels'])
x_train, y_train = get_x_labels(raw_train_data, labels_to_idxs)
x_test, y_test = get_x_labels(raw_test_data, labels_to_idxs)

"""Convolutional Neural Network (CNN)"""
n_snapshot_frames = 300
x_train, y_train = get_animation_snapshots(x_train, y_train, n_snapshot_frames)
print(x_train.shape, y_train.shape)
x_train = x_train.reshape(x_train.shape[-3] * x_train.shape[0], x_train.shape[-2], x_train.shape[-1], 1)
y_train = y_train.reshape(-1, y_train.shape[-1])
print(x_train.shape, y_train.shape)

"""Create CNN model """
model = create_cnn_model(n_snapshot_frames=x_train.shape[1],
                         n_frame_features=x_train.shape[2],
                         n_labels=len(labels_to_idxs))

"""Train CNN model"""
loss = model.fit(x=x_train, y=y_train,
                 epochs=5, verbose=0)
print(loss.history)

"""Divide test data into snapshots as with training animations"""
x_test, y_test = get_animation_snapshots(x_test, y_test, n_snapshot_frames)
x_test.shape, y_test.shape
accuracy, perplexity = evaluate_prediction(model, 
                                           x_test.reshape(x_test.shape[-3] * x_test.shape[0], x_test.shape[-2], x_test.shape[-1], 1), 
                                           y_test.reshape(-1, y_test.shape[-1]))
print(accuracy, perplexity)

test_animation = x_test[0]
test_animation = test_animation.reshape(x_test.shape[-3], x_test.shape[-2], x_test.shape[-1], 1)
test_animation_labels = y_test[0]
pred_labels, pred_probs = predict_labels(model, test_animation, n_best=1)

for each in list(zip([[idxs_to_labels[label_idx] for label_idx in labels] for labels in pred_labels],
         pred_probs,
         [[idxs_to_labels[label_idx] for label_idx in labels] for labels in test_animation_labels])):
    print(each)
