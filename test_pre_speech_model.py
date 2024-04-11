"""
This model tries to detect the start of speech given EEG signal. The output is the sample at which speech is predicted to start.
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
from helpers.model import create_conv_model_regression, create_conv_model
from helpers.vad import get_speech_marking_for_file, VAD_SAMPLING_RATE, read_audio

from scipy.io import wavfile

from matplotlib.pyplot import figure
plt.rcParams["figure.figsize"] = (7,7)


import os

import random

random.seed(42)

def load_data(speech_type: str, eeg_nodes: list[str], targets: list[str], excluded_subjects=[], num_subjects=20, epochs=20, test_split=0.2, use_filter=True, use_all_nodes=True, only_prespeech=True, speech_detection_model=None):
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
                if speech_type == "overt" and only_prespeech:
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

                if np.any(np.abs(numpy_df) > 40):
                    continue # skip peaky things
                
                # numpy_df = normalize_df(numpy_df)

                if use_filter:
                    numpy_df = filter_data(numpy_df)

                if only_prespeech:
                    try:
                        audio_marking = np.zeros_like(numpy_df[:, 0])

                        if speech_type == "overt":
                            start_speech = markings[0]['start'] / VAD_SAMPLING_RATE * 1024.0
                            start_speech_VAD = int(start_speech)
                            start_speech_model = int(speech_detection_model.predict(np.array([numpy_df]), verbose=0)[0][0] * 2048.0)
                            print(start_speech_VAD, start_speech_model)
                            samplerate, audio = wavfile.read(audio_path)
                            audio = audio.astype(np.float64)
                            audio /= np.max(np.abs(audio))
                            
                            audio_marking[:start_speech_VAD] = 1
                            numpy_df_rm = delete_speech(numpy_df, audio_marking)

                            xs = np.linspace(0, 2, len(numpy_df))
                            xs_audio =  np.linspace(0, 2, len(audio))
                            
                            fig, (eeg_axs, audio_axs) = plt.subplots(2)

                            # fig.suptitle(f"Pre-speech only EEG Segment")
                            eeg_axs.plot(xs, numpy_df, linewidth=0.5, color="black")

                            # eeg_axs.title("EEG data")
                            eeg_axs.axvline(x = start_speech_VAD / 1024, color = 'b', label = 'Onset of speech')
                            # eeg_axs.axvline(x = start_speech_model / 1024, color = 'r', label = 'EEG based detection')

                            eeg_axs.set_xlabel("Sec")
                            eeg_axs.set_title("EEG Data")
                            eeg_axs.set_ylabel("mV")
                            eeg_axs.legend()

                            # # audio_axs.title("Audio data")
                            audio_axs.plot(xs_audio, audio, linewidth=0.5, color="black")
                            audio_axs.set_xlabel("Sec")
                            audio_axs.set_title("Audio Data")

                            audio_axs.axvline(x = start_speech_VAD / 1024, color = 'b', label = 'Onset of speech')
                            # audio_axs.axvline(x = start_speech_model / 1024, color = 'r', label = 'EEG based detection')
                            audio_axs.legend()
                            # plt.subplots_adjust(left=0.084, bottom=0.063, right=0.975,top=0.966, wspace =0, hspace=0.274)
                            plt.show()
                            continue

                        audio_marking[:start_speech] = 1
                        numpy_df = delete_speech(numpy_df, audio_marking)
                        
                    except Exception as e:
                        print(e)
                        exit(0)
                        # skip this datapoint if no speech is detected
                        continue
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



if __name__ == "__main__":
    try:
        os.remove("model_save_prespeech_detection.h5")
    except:
        pass

    try:
        import sys
        experiment_id = int(sys.argv[1])
    except:
        print("No channelset selected")
        exit(1)

    TARGETS = ["aa", "oo", "ee", "ie", "oe"]

    EEG_NODES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    EEG_NODES2 = ["F7", "F5", "FT7", "FC5", "FC3", "FC1", "T7", "C5", "C3", "Cz", "C4", "TP7", "CP5", "CP3", "P5", "P3"]
    EEG_NODES_IOANNIS = ["F3", "F4", "C3", "C4", "P3", "P4"]

    TEST_SPLIT = 0.2
    NUM_SUBJECTS = 20
    EPOCHS = 150
    N_REPEATS = 5

    noise = [9, 13, 7, 17, 2]

    channel_sets = [EEG_NODES_IOANNIS, EEG_NODES, EEG_NODES2]
    prespeech_models = [
        f"./prespeech_final_results/models/model_0_run_0.h5",
        f"./prespeech_final_results/models/model_1_run_2.h5",
        f"./prespeech_final_results/models/model_2_run_2.h5",
        f"./prespeech_final_results/models/model_3_run_0.h5"]

    speech_detection_model = load_model(prespeech_models[experiment_id])

    use_all_nodes = False
    try:
        channel_set_to_use = channel_sets[experiment_id]
    except:
        channel_set_to_use = []
        use_all_nodes = True


    data_pool = {}

    filename=  f"./final_final_results/result_{experiment_id}"
    for run_id in range(N_REPEATS):
        print("LOADING DATA...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            overt_train_x, overt_train_y, overt_test_x, overt_test_y = load_data("overt", channel_set_to_use, TARGETS, excluded_subjects=noise, num_subjects=NUM_SUBJECTS, test_split=TEST_SPLIT, use_filter=True, use_all_nodes=use_all_nodes, only_prespeech=True, speech_detection_model=speech_detection_model)

        print("DONE LOADING DATA")