"""
This module runs an experiment to compare the effect of the speech data in the EEG signal.

First a model is made with all EEG present, this is repeated 3 times and an average accuracy is taken.
After this, a model is made with speech data in which at the start of the speech the EEG data is deleted, so only
prespeech data is available for the model to train on. This is also repeated 3 times and an average accuracy is given.

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
from helpers.model import create_conv_model
from helpers.vad import get_speech_marking_for_file, VAD_SAMPLING_RATE

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

                        audio_marking[start_speech:] = 1 # inverted selection now

                        # numpy_df = delete_speech(numpy_df, audio_marking) 

                        plt.plot(numpy_df)
                        plt.show()
                        
                    except:
                        # skip this datapoint if no speech is detected
                        continue
                exit()
                numpy_df = rereference(numpy_df)
                
                y = [0 for _ in targets]
                y[i] = 1

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




def normalize_coeffs(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return mean, std

if __name__ == "__main__":
    
    print("RUNNING PRE SPEECH TEST SUITE")

    TARGETS = ["aa", "oo", "ee", "ie", "oe"]
    # TARGETS = ["aa", "oo", "ee"]
    EEG_NODES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    EEG_NODES2 = ["F7", "F5", "FT7", "FC5", "FC3", "FC1", "T7", "C5", "C3", "Cz", "C4", "TP7", "CP5", "CP3", "P5", "P3"]
    EEG_NODES_IOANNIS = ["F3", "F4", "C3", "C4", "P3", "P4"]
    SPEECH_TYPE = "overt"

    N_REPEATS = 3

    females = [1, 3, 4, 5, 6, 7, 8, 9, 10, 14, 16, 17, 18, 19]
    noise = [9, 13, 7, 17, 2]

    BASE_LINE_RESULTS = []
    TEST_RESULT = []
    
    print("----------------- RUNNING BASELINE MODEL -------------------")
    for i in range(N_REPEATS):
        print(f"RUN {i + 1}/{N_REPEATS}")
        print("LOADING DATA...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_x, train_y, test_x, test_y = load_data(SPEECH_TYPE, EEG_NODES2, TARGETS, excluded_subjects=noise, num_subjects=20, test_split=0.15, use_filter=True, use_marking=True)

        print("DONE LOADING DATA")
        # create a new model fo this run
        model = create_conv_model(input_shape=(2048, train_x.shape[2]), num_y=train_y.shape[-1])
        model.summary()

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"model_save_baseline_{i}.h5",
            save_weights_only=False,
            monitor='val_loss',
            mode='max',
            save_best_only=True)

        model.fit(train_x, train_y, verbose=2, epochs=150, shuffle=True, batch_size=64,callbacks=[model_checkpoint_callback], validation_data=(test_x, test_y))
        model.save("model.h5")

        model = load_model(f"model_save_baseline_{i}.h5")

        test_loss, test_acc = model.evaluate(test_x, test_y)

        print(test_loss, test_acc)

        results = model.predict(test_x)
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

        BASE_LINE_RESULTS.append(right / total)

    
    print("--------------- BASELINE DONE ---------------")
    print(f"Average accuracy: {sum(BASE_LINE_RESULTS)/len(BASE_LINE_RESULTS)* 100:.2f}%")

    print("----------------- RUNNING TEST MODEL -------------------")
    for i in range(N_REPEATS):
        print(f"RUN {i + 1}/{N_REPEATS}")
        print("LOADING DATA...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_x, train_y, test_x, test_y = load_data(SPEECH_TYPE, EEG_NODES2, TARGETS, excluded_subjects=noise, num_subjects=20, test_split=0.15, use_filter=True, use_marking=True)

        print("DONE LOADING DATA")
        # create a new model fo this run
        model = create_conv_model(input_shape=(2048, train_x.shape[2]), num_y=train_y.shape[-1])
        model.summary()

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"model_save_test_{i}.h5",
            save_weights_only=False,
            monitor='val_loss',
            mode='max',
            save_best_only=True)

        model.fit(train_x, train_y, verbose=2, epochs=150, shuffle=True, batch_size=64,callbacks=[model_checkpoint_callback], validation_data=(test_x, test_y))
        model.save("model.h5")

        model = load_model(f"model_save_test_{i}.h5")

        test_loss, test_acc = model.evaluate(test_x, test_y)

        print(test_loss, test_acc)

        results = model.predict(test_x)
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

        TEST_RESULT.append(right / total)

    
    print("--------------- TEST RESULT DONE ---------------")
    print(f"Average accuracy: {sum(TEST_RESULT)/len(TEST_RESULT)* 100:.2f}%")

    print("------------------------------------------------")
    print()
    print("Baseline", BASE_LINE_RESULTS)
    print("Without speech", TEST_RESULT)