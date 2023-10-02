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

                        numpy_df = delete_speech(numpy_df, audio_marking) 
                    except:
                        # skip this datapoint if no speech is detected
                        continue

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

def run_experiment(name: str, nodes: list, epochs: int , num_subjects: int, num_repeat: int, use_all_nodes=False):
    """
    Runs 1 set of experiments
    """
    RESULT = []

    noise = [9, 13, 7, 17, 2]
    SPEECH_TYPE = "overt"
    TARGETS = ["aa", "oo", "ee", "ie", "oe"]

    print(f"----------------- RUNNING {name} EXPERIMENTS -------------------")
    for i in range(num_repeat):
        try:
            os.remove(f"model_save_{name}_{i}.h5")
        except:
            pass

        print(f"RUN {i + 1}/{num_repeat}")
        print("LOADING DATA...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_x, train_y, test_x, test_y = load_data(SPEECH_TYPE, nodes, TARGETS, excluded_subjects=noise, num_subjects=num_subjects, test_split=0.15, use_filter=True, use_marking=False, use_all_nodes=use_all_nodes)

        print("DONE LOADING DATA")
        # create a new model fo this run
        model = create_conv_model(input_shape=(2048, train_x.shape[2]), num_y=train_y.shape[-1])
        model.summary()

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"model_save_{name}_{i}.h5",
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        model.fit(train_x, train_y, verbose=2, epochs=EPOCHS, shuffle=True, batch_size=64,callbacks=[model_checkpoint_callback], validation_data=(test_x, test_y))
        model.save("model.h5")

        model = load_model(f"model_save_{name}_{i}.h5")

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

        RESULT.append(right / total)

    print(f"--------------- {name} RESULT DONE ---------------")
    print(f"Average accuracy: {sum(RESULT)/len(RESULT)* 100:.2f}%")

    return RESULT

if __name__ == "__main__":

    EPOCHS = 150
    NUM_SUBJECTS = 20
    
    print("RUNNING CHANNEL EXPERIMENT SUITE")

    PAPER_NODES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    POSTER_NODES = ["F7", "F5", "FT7", "FC5", "FC3", "FC1", "T7", "C5", "C3", "Cz", "C4", "TP7", "CP5", "CP3", "P5", "P3"]
    IOANNIS_NODES = ["F3", "F4", "C3", "C4", "P3", "P4"]

    N_REPEATS = 3

    ALL_NODES_RESULT = run_experiment("ALL_NODES", [], epochs=EPOCHS, num_repeat=N_REPEATS, num_subjects=NUM_SUBJECTS, use_all_nodes=True)
    IOANNIS_NODES_RESULT = run_experiment("IOANNIS_NODES", IOANNIS_NODES, epochs=EPOCHS, num_repeat=N_REPEATS, num_subjects=NUM_SUBJECTS, use_all_nodes=False)
    POSTER_NODES_RESULT =  run_experiment("POSTER_NODES", POSTER_NODES, epochs=EPOCHS, num_repeat=N_REPEATS, num_subjects=NUM_SUBJECTS, use_all_nodes=False)
    PAPER_NODES_RESULT =  run_experiment("PAPER_NODES", PAPER_NODES, epochs=EPOCHS, num_repeat=N_REPEATS, num_subjects=NUM_SUBJECTS, use_all_nodes=False)
    
   
    print("------------------------------------------------")
    print()
    print("Baseline", ALL_NODES_RESULT , sum(ALL_NODES_RESULT) / len(ALL_NODES_RESULT))
    print("Ioannis", IOANNIS_NODES_RESULT , sum(IOANNIS_NODES_RESULT) / len(IOANNIS_NODES_RESULT))
    print("Poster", POSTER_NODES_RESULT , sum(POSTER_NODES_RESULT) / len(POSTER_NODES_RESULT))
    print("Paper", PAPER_NODES_RESULT , sum(PAPER_NODES_RESULT) / len(PAPER_NODES_RESULT))