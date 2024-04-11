"""
This experiment compares the performance of speech start recognition from EEG signals.

A CNN is used with a regression target being the start sample of speech. The models is trained on all subjects, 
and trained for each subject seperately to compare the performance.

"""

from mne.io import read_epochs_eeglab
import numpy as np
import matplotlib.pyplot as plt
import warnings
from librosa import resample
import tensorflow as tf

from keras.models import load_model

from helpers.filtering import filter_data, rereference
from helpers.preprocessing import delete_speech
from helpers.model import create_conv_model_regression
from helpers.vad import get_speech_marking_for_file, VAD_SAMPLING_RATE

import os

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

                numpy_df = rereference(numpy_df)
                
                try:
                    y = markings[0]['start'] / VAD_SAMPLING_RATE * 1024.0
                    y/= 2048.0
                except:
                    # skip sample
                    continue

                if epoch in test_array: # this now is a test sample :)
                    test_xs.append(numpy_df)
                    test_ys.append(np.array(y))
                else:    
                    xs.append(numpy_df)
                    ys.append(np.array(y))

            loaded += 1
            print(f"Loaded {loaded}/{total_data_count} data", end="\r")
    print()
    return np.array(xs), np.array(ys).reshape((-1, 1)), np.array(test_xs), np.array(test_ys).reshape((-1, 1))


def run_experiment(name: str, N_REPEATS=3):
    try:
        os.remove(f"model_save_prespeech_detection_{name}.h5")
    except:
        pass

    TARGETS = ["aa", "oo", "ee", "ie", "oe"]
    EEG_NODES2 = ["F7", "F5", "FT7", "FC5", "FC3", "FC1", "T7", "C5", "C3", "Cz", "C4", "TP7", "CP5", "CP3", "P5", "P3"]
    SPEECH_TYPE = "overt"

    N_REPEATS = 3

    females = [1, 3, 4, 5, 6, 7, 8, 9, 10, 14, 16, 17, 18, 19]
    noise = [9, 13, 7, 17, 2]

    BASE_LINE_RESULTS = []
    TEST_RESULT = []

    print("LOADING DATA...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_x, train_y, test_x, test_y = load_data(SPEECH_TYPE, EEG_NODES2, TARGETS, excluded_subjects=noise, num_subjects=1, test_split=0.15, use_filter=True, use_marking=True)

    print("DONE LOADING DATA")
    # create a new model fo this run
    model = create_conv_model_regression(input_shape=(2048, train_x.shape[2]), num_y=train_y.shape[-1])
    model.summary()

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"model_save_prespeech_detection_{name}.h5",
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    model.fit(train_x, train_y, verbose=2, epochs=150, shuffle=True, batch_size=16,callbacks=[model_checkpoint_callback], validation_data=(test_x, test_y))
    model.save("model.h5")


if __name__ == "__main__":
    run_experiment("Testrun")