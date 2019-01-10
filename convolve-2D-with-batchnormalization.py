# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:43:55 2018

@author: shree
"""
import os, json
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import keras.utils
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix

rowsPerInstance = 600
colsPerInstance = 6
labelCountDict = defaultdict(int)

# ***************************************************##***************************************************#
def getLabels():
    AllCharacterMotion = ['arrive', 'bust', 'circearound', 'close', 'depart', 'enter', 'exit', 'flinch',
                          'halfclose', 'halfopen', 'meander', 'open', 'scale', 'shake', 'shuffle', 'spin',
                         'approach', 'encircle', 'hit', 'miss', 'nod', 'push', 'stopped']
    OneCharacterMotion = ['arrive', 'bust', 'circearound', 'close', 'depart', 'enter', 'exit', 'flinch',
                          'halfclose', 'halfopen', 'meander', 'open', 'scale', 'shake', 'shuffle', 'spin']
    TwoCharacterMotion = ['approach', 'encircle', 'hit', 'miss', 'nod', 'push', 'stopped']
    ##touch not included

    dictLabel_to_Idx = {k: v for v, k in enumerate(AllCharacterMotion)}
    dictIdx_to_Label = {v: k for k, v in dictLabel_to_Idx.items()}

    return dictLabel_to_Idx, dictIdx_to_Label


# ***************************************************##***************************************************#
def requiredLabel(label, labels_to_idxs):
    for k, v in labels_to_idxs.items():
        if k in label:
            return True
    return False

# ***************************************************##***************************************************#
def find_agent(currLabel):
    if currLabel.count('-') == 0:
        return []
    elif currLabel.count('-') == 1:
        return [currLabel.split('-')[0]]
    else:
        return [currLabel.split('-')[0], currLabel.split('-')[-1]]


# ***************************************************##***************************************************#
def find_train_instances(animation_files, labels_to_idxs):
    dict_agent_to_column = {}
    dict_agent_to_column['BT'] = [1, 4]
    dict_agent_to_column['LT'] = [4, 7]
    dict_agent_to_column['C'] = [7, 10]
    #    dict_agent_to_column['D'] = [11]

    padZeroes = [0] * colsPerInstance
    countData = 0
    dictData = defaultdict(dict)

    for animation_file in animation_files:
        with open(animation_file, 'r') as f:
            animation = json.load(f)

        animation = np.array(animation)
        index = 0
        while index < len(animation):
            curr_data = animation[index]
            if not requiredLabel(curr_data[0], labels_to_idxs):
                index += 1
            else:
                instance = []
                current_label = curr_data[0]

                startindex = index
                while index < len(animation) and animation[index][0] == current_label:
                    index += 1
                endindex = index

                current_agents = find_agent(curr_data[0])
                ZeroColumns = np.zeros((endindex - startindex, 3))
                if current_agents == []:
                    instance = ZeroColumns[:]
                else:
                    columns_to_fetch = dict_agent_to_column[current_agents[0]]
                    instance = animation[startindex:endindex, columns_to_fetch[0]:columns_to_fetch[1]]

                if len(current_agents) == 2:
                    columns_to_fetch = dict_agent_to_column[current_agents[1]]
                    instance = np.hstack(
                        (instance, animation[startindex:endindex, columns_to_fetch[0]:columns_to_fetch[1]]))
                else:
                    instance = np.hstack((instance, ZeroColumns))

                while len(instance) < rowsPerInstance:
                    instance = np.vstack((instance, padZeroes))
                if len(instance) > rowsPerInstance:
                    instance = instance[-rowsPerInstance:]

                if current_label.count('-') == 0:
                    current_label = 'stopped'
                    dictData[countData]['label'] = labels_to_idxs[current_label]
                else:
                    current_label = current_label.split('-')[1]
                    dictData[countData]['label'] = labels_to_idxs[current_label]
                dictData[countData]['data'] = instance
                countData += 1

                if current_label in labelCountDict:
                    labelCountDict[current_label] += 1
                else:
                    labelCountDict[current_label] = 1

    animation_data = pd.DataFrame.from_dict(dictData, orient='index')

    return animation_data


# ***************************************************##***************************************************#
def data_reshape(x_train, y_train, num_category):
    x_train = x_train.reshape(x_train.shape[0], rowsPerInstance, colsPerInstance, 1)
    print('X shape:', x_train.shape)

    """convert class vectors to binary class matrices"""
    y_train = keras.utils.np_utils.to_categorical(y_train, num_category)
    print('y shape:', y_train.shape)

    return x_train, y_train



# ***************************************************##***************************************************#
def cnn_model_building(input_shape, num_category):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(4, 2),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (4, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (4, 2), activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(num_category, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print(model.summary())
    return model


# ***************************************************##***************************************************#
def rnn_model_building(input_shape, num_category):

    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=input_shape))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))  # return a single vector of dimension 32
    model.add(Dense(num_category, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print(model.summary())

    return model

# ***************************************************##***************************************************#
def model_training(model, x_train, y_train, x_test, y_test):
    batch_size = 32
    num_epoch = 100
    model_log = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=num_epoch,
                          verbose=1,
                          validation_data=(x_test, y_test))

    # summarize history for accuracy
    plt.plot(model_log.history['acc'])
    plt.plot(model_log.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    figtosave = plt.gcf()
    plt.show()
    figtosave.savefig('accuracyPlot')

    # summarize history for loss
    plt.plot(model_log.history['loss'])
    plt.plot(model_log.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    figtosave = plt.gcf()
    plt.show()
    figtosave.savefig('lossPlot')

    return model_log


# ***************************************************##***************************************************#
def model_evaluation(model, x_test, y_test):
    print(x_test.shape, y_test.shape)
    score = model.evaluate(x_test, y_test, verbose=0)
    return score

# ***************************************************##***************************************************#
def model_prediction(model, x_test, test_animation_labels):
    n_best = 2
    all_best_labels = []
    all_best_probs = []
    pred_probs = model.predict(x_test)
    for probs in pred_probs:
        best_labels = np.argsort(probs)[::-1][:n_best]
        all_best_labels.append(best_labels)
        best_probs = probs[best_labels]
        all_best_probs.append(best_probs)
    all_best_labels = np.array(all_best_labels)
    all_best_probs = np.array(all_best_probs)

    pred_labels = all_best_labels
    pred_probs = all_best_probs

    for each in list(zip([[idxs_to_labels[label_idx] for label_idx in labels] for labels in pred_labels],
                         pred_probs,
                         [[idxs_to_labels[label_idx] for label_idx in labels] for labels in test_animation_labels])):
        print(each)


# ***************************************************##***************************************************#

if __name__ == "__main__":

    # ***************************************************##***************************************************#
    animation_filenames = ['allDataFiles/' + filename for filename in os.listdir('allDataFiles')]
    labels_to_idxs, idxs_to_labels = getLabels()
    n_test_animations = 1

    """Training data preprocessing"""  ##Comment this section if training already done

    raw_train_data = find_train_instances(animation_filenames[:-n_test_animations], labels_to_idxs)
    from collections import OrderedDict
    d_sorted_by_value = OrderedDict(sorted(labelCountDict.items(), key=lambda x: x[0]))
    for k, v in d_sorted_by_value.items():
        print("%s: %s" % (k, v))
    y_train = np.vstack(raw_train_data['label'].values)
    x_train = np.stack(raw_train_data['data'].values)

    raw_test_data = find_train_instances(animation_filenames[-n_test_animations:], labels_to_idxs)
    y_test = np.vstack(raw_test_data['label'].values)
    x_test = np.stack(raw_test_data['data'].values)

    num_category = len(labels_to_idxs)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=0, test_size=0.20)
    x_train, y_train = data_reshape(x_train, y_train, num_category)
    x_val, y_val = data_reshape(x_val, y_val, num_category)
    test_labels = y_test
    x_test, y_test = data_reshape(x_test, y_test, num_category)

    """Create a Model, train it and save to the disk"""   ##Comment this section if model already saved

    ##### CNN ###
    model = cnn_model_building(input_shape=(rowsPerInstance, colsPerInstance, 1), num_category = num_category)
    model_log = model_training(model, x_train, y_train, x_val, y_val)
    #############


    ##### RNN ###
    # x_train = x_train.reshape(x_train.shape[0], rowsPerInstance, colsPerInstance)
    # x_val = x_val.reshape(x_val.shape[0], rowsPerInstance, colsPerInstance)
    # x_test = x_test.reshape(x_test.shape[0], rowsPerInstance, colsPerInstance)
    # model = rnn_model_building(input_shape=(rowsPerInstance, colsPerInstance), num_category=len(labels_to_idxs))
    # model_log = model_training(model, x_train, y_train, x_val, y_val)
    #############

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


    """Testing Phase: Read the model saved in previous step and test on new data"""
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss=keras.losses.categorical_crossentropy,
                         optimizer=keras.optimizers.Adadelta(),
                         metrics=['accuracy'])

    model_prediction(loaded_model, x_test, test_labels)
    # score = model_evaluation(model, x_test, test_animation_labels)
    # print("score", score)
