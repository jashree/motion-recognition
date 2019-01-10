import os, json
from collections import defaultdict
import pandas as pd
import numpy as np
import keras
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

modelType = 'CNN'                     # 'CNN' or 'SVM'
rowsPerInstance = 600
colsPerInstance = 3
labelCountDict = defaultdict(int)


# ***************************************************##***************************************************#
def getLabels():
    # OneCharacterMotion = ['arrive', 'bust', 'circearound', 'close', 'depart', 'enter', 'exit', 'flinch',
    #                      'halfclose', 'halfopen', 'meander', 'open', 'scale', 'shake', 'shuffle', 'spin']
    OneCharacterMotion = ['meander', 'bust', 'shuffle', 'flinch']

    dictLabel_to_Idx = {k: v for v, k in enumerate(OneCharacterMotion)}
    dictIdx_to_Label = {v: k for k, v in dictLabel_to_Idx.items()}

    return dictLabel_to_Idx, dictIdx_to_Label


# ***************************************************##***************************************************#
def find_instances(animation_files, labels_to_idxs):
    dict_agent_to_column = {}
    dict_agent_to_column['BT'] = [1, 4]
    dict_agent_to_column['LT'] = [4, 7]
    dict_agent_to_column['C'] = [7, 10]
    padZeroes = [0] * colsPerInstance
    countData = 0
    dictData = defaultdict(dict)
    for animation_file in animation_files:
        with open(animation_file, 'r') as f:
            animation = json.load(f)
        index = 0
        while index < len(animation):
            curr_data = animation[index]
            if requiredLabel(curr_data[0], labels_to_idxs):
                current_label = curr_data[0]
                current_agent = find_agent(curr_data[0])
                instance = []
                while animation[index][0] == current_label:
                    curr_data = animation[index]
                    columns_to_fetch = dict_agent_to_column[current_agent[0]]
                    instance.append(curr_data[columns_to_fetch[0]:columns_to_fetch[1]])
                    if index < len(animation) - 1:
                        index += 1
                    else:
                        break
                while len(instance) < rowsPerInstance:
                    instance.append(padZeroes)
                if len(instance) > rowsPerInstance:
                    instance = instance[-rowsPerInstance:]

                dictData[countData]['label'] = labels_to_idxs[curr_data[0].split('-')[1]]
                dictData[countData]['data'] = instance
                countData += 1

                if curr_data[0].split('-')[1] in labelCountDict:
                    labelCountDict[curr_data[0].split('-')[1]] += 1
                else:
                    labelCountDict[curr_data[0].split('-')[1]] = 1

            index += 1

    animation_data = pd.DataFrame.from_dict(dictData, orient='index')

    return animation_data


# ***************************************************##***************************************************#
def requiredLabel(label, labels_to_idxs):
    for k, v in labels_to_idxs.items():
        if k in label:
            return True
    return False


# ***************************************************##***************************************************#
def find_agent(currLabel):
    if currLabel.count('-') == 1:
        return [currLabel.split('-')[0]]


# ***************************************************##***************************************************#
def cnn_model_data_reshape(x_train, x_test, y_train, y_test, num_category):
    x_train = x_train.reshape(x_train.shape[0], rowsPerInstance, colsPerInstance, 1)
    x_test = x_test.reshape(x_test.shape[0], rowsPerInstance, colsPerInstance, 1)
    input_shape = (rowsPerInstance, colsPerInstance, 1)

    print('X_train shape:', x_train.shape)
    print('X_test shape:', x_test.shape)

    """convert class vectors to binary class matrices"""
    print(y_train.shape)
    print(y_test.shape)

    y_train = keras.utils.to_categorical(y_train, num_category)
    y_test = keras.utils.to_categorical(y_test, num_category)

    print('y_train shape:', y_train.shape)
    print('y_test_shape:', y_test.shape)

    return x_train, y_train, x_test, y_test, input_shape


# ***************************************************##***************************************************#
def cnn_model_building(input_shape, num_category):
    model = Sequential()
    # convolutional layer with rectified linear unit activation
    model.add(Conv2D(32, kernel_size=(2, 2),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    model.add(Flatten())

    model.add(Dense(800, activation='sigmoid'))

    model.add(Dropout(0.5))
    model.add(Dense(800, activation='sigmoid'))

    model.add(Dense(num_category, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.adam(),
                  metrics=['accuracy'])
    print(model.summary())
    return model


# ***************************************************##***************************************************#
def cnn_model_training(model, x_train, y_train, x_test, y_test):
    batch_size = 5
    num_epoch = 50
    model_log = model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=num_epoch,
                          verbose=1,
                          validation_data=(x_test, y_test))
    return model_log


# ***************************************************##***************************************************#
def cnn_model_evaluation(model, model_log, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)

    # summarize history for accuracy
    plt.plot(model_log.history['acc'])
    plt.plot(model_log.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(model_log.history['loss'])
    plt.plot(model_log.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return score


# ***************************************************##***************************************************#
def cnn_model_prediction(model, x_test, test_animation_labels):
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
def plot_motions(x_train, y_train, idxs_to_labels, labels_to_idxs, number_to_plot=2):
    for eachmotion in idxs_to_labels:
        fig, ax = plt.subplots(number_to_plot, 3, sharey=False)
        num = 0
        for i in range(0, len(x_train)):
            if y_train[i, 0] == eachmotion:

                for ntype in range(0, 3):

                    """Plot x-value v/s y-value"""
                    if ntype == 0:
                        X_to_plot = x_train[i, :, 0]
                        Y_to_plot = x_train[i, :, 1]

                        ax[num, ntype].plot(X_to_plot, Y_to_plot)
                        ax[num, ntype].set_yticks([])
                        ax[num, ntype].set_xticks([])

                    """Plot x-value, y-value and r-value v/s time"""
                    if ntype == 1:
                        X_to_plot = x_train[i, :, 0]
                        Y_to_plot = x_train[i, :, 1]
                        R_to_plot = x_train[i, :, 2]

                        ax[num, ntype].plot(X_to_plot, 'r-', label='X value')
                        ax[num, ntype].plot(Y_to_plot, 'b-', label='Y value')
                        ax[num, ntype].plot(R_to_plot, 'g-', label='R value')
                        ax[num, ntype].set_yticks([])
                        ax[num, ntype].set_xticks([])

                    """Plot x-value, y-value and r-value v/s time using 3 different y-axis"""
                    if ntype == 2:
                        ax[num, ntype].plot(X_to_plot, 'r-', label='X value')

                        ax2 = ax[num, ntype].twinx()
                        ax2.plot(Y_to_plot, 'b-', label='Y value')
                        ax2.set_yticks([])

                        ax3 = ax[num, ntype].twinx()
                        ax3.plot(R_to_plot, 'g-', label='R value')
                        ax3.set_yticks([])

                        ax[num, ntype].set_yticks([])
                        ax[num, ntype].set_xticks([])

                num += 1
                if num == number_to_plot:
                    break
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle(idxs_to_labels[eachmotion])
        #        figtosave = plt.gcf()
        plt.show()


#        figtosave.savefig(idxs_to_labels[eachmotion] + str(i) +'.png')

# ***************************************************##***************************************************#
def svc_model_data_reshape(x_train, x_test, y_train, y_test, num_category):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    y_train = y_train.ravel()
    print('X-train shape: ', x_train.shape)
    print('Y-train shape: ', y_train.shape)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    y_test = y_test.ravel()
    print('X-test shape: ', x_test.shape)
    print('Y-test shape: ', y_test.shape)
    return x_train, y_train, x_test, y_test


# ***************************************************##***************************************************#
def svc_model_training(X, Y):
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    clf.fit(X, Y)
    print(clf)
    return clf


# ***************************************************##***************************************************#
def svc_model_prediction(clf, X, Y, idxs_to_labels):
    pred = clf.predict(X)
    print('Predicted : Expected')
    for i, y in enumerate(Y):
        print(idxs_to_labels[pred[i]], idxs_to_labels[Y[i]])
    return pred


# ***************************************************##***************************************************#
def svc_model_evaluation(clf, pred, X, Y):
    score = clf.score(X, Y)
    cm = confusion_matrix(y_test, pred)
    print('Accuracy: ', score)
    print('Confusion Matrix: \n', cm)
    return score, cm


# ***************************************************##***************************************************#
if __name__ == "__main__":
    animation_filenames = ['allDataFiles/' + filename for filename in os.listdir('allDataFiles')]
    #    n_test_animations = 1
    labels_to_idxs, idxs_to_labels = getLabels()

    """Segregate test and training data"""
    raw_train_data = find_instances(animation_filenames[:], labels_to_idxs)
    #    raw_train_data = find_instances(animation_filenames[:-n_test_animations], labels_to_idxs)
    #    raw_test_data = find_instances(animation_filenames[-n_test_animations:], labels_to_idxs)

    from collections import OrderedDict

    d_sorted_by_value = OrderedDict(sorted(labelCountDict.items(), key=lambda x: x[1]))
    for k, v in d_sorted_by_value.items():
        print("%s: %s" % (k, v))

    """Divide data between features(x) and corresponding labels(y)"""
    y_train = np.vstack(raw_train_data['label'].values)
    x_train = np.stack(raw_train_data['data'].values)

    #    y_test = np.vstack(raw_test_data['label'].values)
    #    x_test = np.stack(raw_test_data['data'].values)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=0, test_size=0.20)
    test_animation_labels = y_test

    """For creating motion plot"""
    #    plot_motions(x_train, y_train, idxs_to_labels, labels_to_idxs)

    """For SVC Model"""
    if modelType == 'SVM':
        x_train, y_train, x_test, y_test = svc_model_data_reshape(x_train, x_test, y_train, y_test, num_category = len(labels_to_idxs))
        clf = svc_model_training(x_train, y_train)
        pred = svc_model_prediction(clf,x_test,y_test, idxs_to_labels)
        score, cm = svc_model_evaluation(clf, pred, x_test, y_test)

    """For CNN Model"""
    if modelType == 'CNN':
        x_train, y_train, x_test, y_test, input_shape = cnn_model_data_reshape(x_train, x_test, y_train, y_test,
                                                                               num_category=len(labels_to_idxs))
        model = cnn_model_building(input_shape, len(labels_to_idxs))
        print(model.summary)
        model_log = cnn_model_training(model, x_train, y_train, x_test, y_test)
        score = cnn_model_evaluation(model, model_log, x_test, y_test)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        cnn_model_prediction(model, x_test, test_animation_labels)
