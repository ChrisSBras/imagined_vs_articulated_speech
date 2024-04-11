import os
from mne.io import read_epochs_eeglab
import numpy as np
import matplotlib.pyplot as plt
import warnings
from librosa import resample
import tensorflow as tf

from keras.models import load_model

from helpers.filtering import filter_data, rereference
from helpers.preprocessing import delete_speech
from helpers.model import create_conv_model
from helpers.vad import get_speech_marking_for_file, VAD_SAMPLING_RATE
from helpers.io import mkdir_p
from sklearn.preprocessing import MinMaxScaler

from helpers.tsne import tsne

import pylab

import random

random.seed(42)

def load_data(speech_type: str, eeg_nodes: list[str], targets: list[str], excluded_subjects=[], num_subjects=20, epochs=20, test_split=0.2, use_filter=False, use_all_nodes=True, use_marking=True):
    xs = []
    ys = []

    test_xs = []
    test_ys = []

    total_data_count = len(targets) * num_subjects
    loaded = 0
    
    for i, target in enumerate(targets):

        for j in range(1, num_subjects + 1): # 20 subjects in total
            
            if j in excluded_subjects: # skip excluded subjects
                continue

            subject = f"sub-{j:02}"
            data = read_epochs_eeglab(f"./data/derivatives/{subject}/eeg/{subject}_task-{speech_type}-{target}_eeg.set", verbose=False)
            df = data.to_data_frame()

            test_array = random.sample(range(epochs), round(test_split * epochs) )

            for epoch in range(epochs):
                
                # get audio data
                if speech_type == "overt" and use_marking:
                    audio_path = f"./data/sourcedata/{subject}/audio/{subject}_task-overt-{target}_run-{(epoch + 1):02d}_audio.wav"
                    markings = get_speech_marking_for_file(audio_path)

                epoch_df = df[df["epoch"] == epoch]
               
                if "FC2" in epoch_df.keys():
                     filtered_df = epoch_df.drop(["time", "condition", "epoch", 'FC2'], axis=1)
                else:
                     filtered_df = epoch_df.drop(["time", "condition", "epoch"], axis=1)

                if not use_all_nodes:
                    filtered_df = epoch_df[eeg_nodes]

                numpy_df = filtered_df.to_numpy()
      
                if numpy_df.shape[0] != 2048:
                    continue # skip epochs that are not exactly right for now

                if use_filter:
                    numpy_df = filter_data(numpy_df)

                if use_marking:
                    try:
                        audio_marking = np.zeros_like(numpy_df[:, 0])
                        start_speech = markings[0]['start'] / VAD_SAMPLING_RATE * 1024.0
                        start_speech = int(start_speech)

                        audio_marking[:start_speech] = 1 # inverted selection now

                        numpy_df = delete_speech(numpy_df, audio_marking) 
                    except:
                        # skip this datapoint if no speech is detected
                        continue

                numpy_df = rereference(numpy_df)
                
                y = [0 for _ in range(len(targets) + 1)] # + 1 for rest class
                y[i] = 1 # targets are 0 indexed, rest is always last class

                if epoch in test_array: # this now is a test sample :)
                    test_xs.append(numpy_df[:])
                    test_ys.append(np.array(y))
                else:    
                    xs.append(numpy_df[:])
                    ys.append(np.array(y))

            loaded += 1
            print(f"Loaded {loaded}/{total_data_count} data", end="\r")
    print()
    return np.array(xs), np.array(ys), np.array(test_xs), np.array(test_ys)

def load_rest_data( eeg_nodes: list[str], excluded_subjects=[], num_subjects=20, epochs=20, test_split=0.2, use_filter=False, use_all_nodes=True, num_targets=6):
    xs = []
    ys = []

    test_xs = []
    test_ys = []

    total_data_count = num_subjects
    loaded = 0

    for j in range(1, num_subjects + 1): # 20 subjects in total
            
            if j in excluded_subjects: # skip excluded subjects
                continue

            subject = f"sub-{j:02}"
            data = read_epochs_eeglab(f"./data/derivatives/{subject}/eeg/{subject}_task-rest_eeg.set", verbose=False)
            df = data.to_data_frame()

            # pick n random epochs to use, to keep data balanced
            epochs_to_use = random.sample(range(df['epoch'].max()), epochs)
            test_array = random.sample(epochs_to_use, round(test_split * epochs))

            for epoch in epochs_to_use:
                epoch_df = df[df["epoch"] == epoch]
               
                if "FC2" in epoch_df.keys():
                     filtered_df = epoch_df.drop(["time", "condition", "epoch", 'FC2'], axis=1)
                else:
                     filtered_df = epoch_df.drop(["time", "condition", "epoch"], axis=1)

                if not use_all_nodes:
                    filtered_df = epoch_df[eeg_nodes]

                numpy_df = filtered_df.to_numpy()
      
                if numpy_df.shape[0] != 2048:
                    continue # skip epochs that are not exactly right for now

                if use_filter:
                    numpy_df = filter_data(numpy_df)

                numpy_df = rereference(numpy_df)
                
                y = [0 for _ in range(num_targets)]
                y[-1] = 1

                if epoch in test_array: # this now is a test sample :)
                    test_xs.append(numpy_df)
                    test_ys.append(np.array(y))
                else:    
                    xs.append(numpy_df)
                    ys.append(np.array(y))

            loaded += 1
            print(f"Loaded {loaded}/{total_data_count} data", end="\r")

    print()
    return np.array(xs), np.array(ys), np.array(test_xs), np.array(test_ys)


def run_analysis(nodes: list, num_subjects: int, use_all_nodes=False):
    TEST_SPLIT = 0.0

    noise = [9, 13, 7, 17, 2]
    SPEECH_TYPE = "overt"
    TARGETS = ["aa", "oo", "ee", "ie", "oe"]

    # TARGETS = ["aa", "oe"]

    colors_choices = ["red", "green", "blue", "yellow", "orange", "black"]

    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_x, train_y, test_x, test_y = load_data(SPEECH_TYPE, nodes, TARGETS, excluded_subjects=noise, num_subjects=num_subjects, test_split=TEST_SPLIT, use_filter=True, use_marking=False, use_all_nodes=use_all_nodes)

    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rest_train_x, rest_train_y, rest_test_x, rest_test_y = load_rest_data(nodes, excluded_subjects=noise, num_subjects=num_subjects, test_split=TEST_SPLIT, epochs=20, use_filter=True, use_all_nodes=use_all_nodes, num_targets=len(TARGETS) + 1)

    train_x = np.vstack((train_x, rest_train_x))
    train_y = np.vstack((train_y, rest_train_y))
    test_x = np.vstack((test_x, rest_test_x))
    test_y = np.vstack((test_y, rest_test_y))
    # Y = tsne(train_x, 3, 50, 20.0)
    # pylab.scatter(Y[:, 0], Y[:, 1], 20, train_y)
    # pylab.show()

    colors = []

    for i in train_y:
        value = np.argmax(i)
        colors.append(colors_choices[value])

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import time

    plt.ion()

    figure, ax = plt.subplots(figsize=(10, 8))

    X_tsne = TSNE().fit_transform(train_x[:, 0, :])
    data, = ax.plot(X_tsne[:, 0], X_tsne[:, 1], color=colors, alpha=.4)
    plt.show()

    x_total = None
    y_total = None
    x = X_tsne[:, 0]
    y = X_tsne[:, 1]
    sc = ax.scatter(x, y, c=colors)

    first = True
    count = 1 
    for i in range(0, 2048, 10):
        X_tsne = TSNE(n_components=2, n_jobs=-1).fit_transform(train_x[:, i, :])

        if first:
            x_total = X_tsne[:, 0].copy()
            y_total = X_tsne[:, 1].copy()
            first = False
        else:
            x_total += X_tsne[:, 0]
            y_total += X_tsne[:, 1]

            x = x_total / count
            y = y_total / count

            # x = X_tsne[:, 0]
            # y = X_tsne[:, 1]

            count += 1

        sc.set_offsets(np.c_[x,y])

        ax.set_title(f"T = {i}")
        plt.xlim([np.min(x), np.max(x)])
        plt.ylim([np.min(y), np.max(y)])
        figure.canvas.draw_idle()
        plt.pause(0.01
                  )
    plt.waitforbuttonpress()

if __name__ == "__main__":
    NUM_SUBJECTS = 1

    NODES = ["F7", "F5", "FT7", "FC5", "FC3", "FC1", "T7", "C5", "C3", "Cz", "C4", "TP7", "CP5", "CP3", "P5", "P3"]
    NODES = []
    run_analysis(NODES,  num_subjects=NUM_SUBJECTS, use_all_nodes=NODES == [])