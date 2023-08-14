from mne.io import read_raw_eeglab, read_epochs_eeglab, RawArray
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
import warnings
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sklearn.ensemble import RandomForestClassifier

from tslearn.svm import TimeSeriesSVC

from sktime.pipeline import make_pipeline
from sktime.transformations.panel.catch22 import Catch22

from sklearn.metrics import accuracy_score

from pyts.classification import KNeighborsClassifier

from librosa import resample

def moving_average(a, n=200):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]


import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Conv1D, Dropout
from keras import regularizers
import keras.layers

import random

random.seed(69)

def load_data(speech_type: str, eeg_nodes: list[str], targets: list[str], excluded_subjects=[], num_subjects=20, epochs=20, test_split=0.2):
    xs = []
    ys = []

    test_xs = []
    test_ys = []
    

    for i, target in enumerate(targets):

        for j in range(1, num_subjects + 1): # 20 subjects in total
            
            if j in excluded_subjects: # skip excluded subjects
                continue

            subject = f"sub-{j:02}"
            data = read_epochs_eeglab(f"./data/derivatives/{subject}/eeg/{subject}_task-{speech_type}-{target}_eeg.set", verbose=False)
            df = data.to_data_frame()

            test_array = random.sample(range(epochs), round(test_split * epochs) )
            print("Using epochs as test: ", test_array)

            for epoch in range(epochs):
                
                epoch_df = df[df["epoch"] == epoch]
               

                if "FC2" in epoch_df.keys():
                     filtered_df = epoch_df.drop(["time", "condition", "epoch", 'FC2'], axis=1)
                else:
                     filtered_df = epoch_df.drop(["time", "condition", "epoch"], axis=1)

                # filtered_df = epoch_df[eeg_nodes]
                numpy_df = filtered_df.to_numpy()

                if numpy_df.shape[0] != 2048:
                    continue # skip epochs that are not exactly right for now
                
                # new_array = []
                        
                # for data in numpy_df.T:
                #     new_array.append(moving_average(data.copy()))

                # numpy_df = np.array(new_array).T

                y = [0 for _ in targets]
                y[i] = 1

                # y = i
                # y = [i / len(targets)]

                if epoch in test_array: # this now is a test sample :)
                    test_xs.append(numpy_df)
                    test_ys.append(np.array(y))
                else:    
                    xs.append(numpy_df)
                    ys.append(np.array(y))
        
    return np.array(xs), np.array(ys), np.array(test_xs), np.array(test_ys)

def create_model():
    model = Sequential()
    # model.add(LSTM(units=16, input_shape=(2048, 8), return_sequences=False, stateful=False))
    model.add(LSTM(48, activity_regularizer=regularizers.L2(1e-5), return_sequences=True ))
    model.add(Dense(48, activation="tanh", activity_regularizer=regularizers.L2(1e-5)))
    model.add(LSTM(16, activity_regularizer=regularizers.L2(1e-5), return_sequences=False ))
    model.add(Dense(4, activation="relu", activity_regularizer=regularizers.L2(1e-5)))
    model.add(Dense(5, activation="softmax"))

    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        name="sparse_categorical_crossentropy",
    )
    cce = tf.keras.losses.CategoricalCrossentropy()

    model.compile(tf.optimizers.Adam(1e-2), loss=cce, run_eagerly=False,  metrics=["accuracy"])
    
    return model

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"model_save.h5",
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

def normalize_coeffs(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return mean, std

def make_model(input_shape=(2048, 61)):
    input_layer = keras.layers.Input(input_shape)

    filter_size = 64
    kernel_size = 3

    conv1 = keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv1 = keras.layers.Dropout(0.25)(conv1)

    conv2 = keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv2 = keras.layers.Dropout(0.25)(conv2)

    conv3 = keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    conv3 = keras.layers.Dropout(0.25)(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(5, activation="softmax")(gap)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    cce = tf.keras.losses.CategoricalCrossentropy()

    model.compile(tf.optimizers.Adam(1e-2), loss=cce, run_eagerly=False,  metrics=["accuracy"])

    return model 

# def filter_data(data):
#     order = 6
#     fs = 1024.0      # sample rate, Hz
#     cutoff = 100 # desired cutoff frequency of the filter, Hz

#     # Get the filter coefficients so we can check its frequency response.
#     b, a = butter_lowpass(cutoff, fs, order)

if __name__ == "__main__":
 
    TARGETS = ["aa", "oo", "ee", "ie", "oe"]
    EEG_NODES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    EEG_NODES2 = ["F7", "F5", "FT7", "FC5", "FC3", "FC1", "T7", "C5", "C3", "Cz", "C4", "TP7", "CP5", "CP3", "P5", "P3"]
    SPEECH_TYPE = "overt"

    females = [1, 3, 4, 5, 6, 7, 8, 9, 10, 14, 16, 17, 18, 19]
    noise = [9, 13, 7, 17, 2]
    

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     train_x, train_y, test_x, test_y = load_data(SPEECH_TYPE, EEG_NODES2, TARGETS, excluded_subjects=noise, num_subjects=20, test_split=0.2)

    # np.save("train_x", train_x)
    # np.save("train_y", train_y)
    # np.save("test_x", test_x)
    # np.save("test_y", test_y)


    train_x = np.load("train_x.npy")
    train_y = np.load("train_y.npy")

    test_x = np.load("test_x.npy")
    test_y = np.load("test_y.npy")

    print(test_y)

    print(test_x.shape, train_x.shape)
    
    # mu, std = normalize_coeffs(np.vstack((train_x, test_x)))

    # train_x = (train_x - mu) / std
    # test_x = (test_x - mu) / std

    # rocket = RocketClassifier(num_kernels=2000)
    # rocket.fit(train_x, train_y.flatten())

    # catch22 = Catch22()
    # randf = RandomForestClassifier(n_estimators=100)
    # pipe = make_pipeline(catch22, randf)

    # svc = TimeSeriesSVC(n_jobs=-1, verbose=1)

    # svc.fit(train_x, train_y.flatten())

    # model = load_model("model.h5")

    model = make_model()
    model.fit(train_x, train_y, verbose=1, epochs=100, shuffle=True, batch_size=64,callbacks=[model_checkpoint_callback], validation_data=(test_x, test_y))
    model.save("model.h5")

    model = load_model("model_save.h5")
    # model.save("model_26percent.h")

    results = model.predict(test_x)


    # score = accuracy_score(test_y, results)
    # print(score)

    # exit()

    total = 0
    right = 0
    wrong = 0

    for i, result in enumerate(results):
        y_hat = np.argmax(result) 
        y_real = np.argmax(test_y[i])
        
        if (y_hat == y_real):
            right += 1
        else:
            wrong += 1
        total += 1

    print(f"predicted {right} / {total} correct, accuracy of {(right / total) * 100:.2f}%")