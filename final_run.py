from mne.io import read_epochs_eeglab
import numpy as np
import matplotlib.pyplot as plt
import warnings
from librosa import resample
import tensorflow as tf

from keras.models import load_model

from helpers.filtering import filter_data, rereference, normalize_df
from helpers.preprocessing import delete_speech
from helpers.model import create_conv_model_regression, create_wavenet_model, create_conv_model
from helpers.vad import get_speech_marking_for_file, VAD_SAMPLING_RATE

import os
import json 

import random

random.seed(43)

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
                            start_speech = int(start_speech)
                        else:
                            start_speech = int(speech_detection_model.predict(np.array([numpy_df]), verbose=0)[0][0] * 2048.0)

                        audio_marking[:start_speech] = 1
                        numpy_df = delete_speech(numpy_df, audio_marking)
                        
                    except Exception as e:
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



def run_dataset(train_x, train_y, test_x, test_y, epochs) -> tuple:
    try:
        os.remove("final_run_checkpoint_model.h5")
    except:
        pass

    model = create_conv_model(input_shape=(2048, train_x.shape[2]), num_y=train_y.shape[-1])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"final_run_checkpoint_model.h5",
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    model.fit(train_x, train_y, verbose=2, epochs=epochs, shuffle=True, batch_size=16,callbacks=[model_checkpoint_callback], validation_data=(test_x, test_y))
    model.save("model.h5")
    model = load_model(f"final_run_checkpoint_model.h5")
    
    test_loss, test_acc = model.evaluate(test_x, test_y)

    return model, test_loss, test_acc


def run_experiment(tag: str, data_pool: dict, train_x, train_y, test_x, test_y, epochs) -> dict:

    if tag not in data_pool.keys():
        data_pool[tag] = {
            "accuracy": [],
            "loss": []
        }

    model, loss, accuracy = run_dataset(train_x, train_y, test_x, test_y, epochs)

    data_pool[tag]["accuracy"].append(accuracy)
    data_pool[tag]["loss"].append(loss)

    return data_pool


def backup_data_pool(data_pool, filename):
    with open(filename, "w") as f:
            f.write(json.dumps(data_pool, indent=2))

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
            overt_train_x, overt_train_y, overt_test_x, overt_test_y = load_data("overt", channel_set_to_use, TARGETS, excluded_subjects=noise, num_subjects=NUM_SUBJECTS, test_split=TEST_SPLIT, use_filter=True, use_all_nodes=use_all_nodes, only_prespeech=False, speech_detection_model=None)
            covert_train_x, covert_train_y, covert_test_x, covert_test_y = load_data("covert", channel_set_to_use, TARGETS, excluded_subjects=noise, num_subjects=NUM_SUBJECTS, test_split=TEST_SPLIT, use_filter=True, use_all_nodes=use_all_nodes, only_prespeech=False, speech_detection_model=speech_detection_model)

            ps_overt_train_x, ps_overt_train_y, ps_overt_test_x, ps_overt_test_y = load_data("overt", channel_set_to_use, TARGETS, excluded_subjects=noise, num_subjects=NUM_SUBJECTS, test_split=TEST_SPLIT, use_filter=True, use_all_nodes=use_all_nodes, only_prespeech=True, speech_detection_model=None)
            ps_covert_train_x, ps_covert_train_y, ps_covert_test_x, ps_covert_test_y = load_data("covert", channel_set_to_use, TARGETS, excluded_subjects=noise, num_subjects=NUM_SUBJECTS, test_split=TEST_SPLIT, use_filter=True, use_all_nodes=use_all_nodes, only_prespeech=True, speech_detection_model=speech_detection_model)


            combined_train_x = np.vstack((overt_train_x, covert_train_x))
            combined_train_y = np.vstack((overt_train_y, covert_train_y))
            combined_test_x = np.vstack((overt_test_x, covert_test_x))
            combined_test_y = np.vstack((overt_test_y, covert_test_y))

            ps_combined_train_x = np.vstack((ps_overt_train_x, ps_covert_train_x))
            ps_combined_train_y = np.vstack((ps_overt_train_y, ps_covert_train_y))
            ps_combined_test_x = np.vstack((ps_overt_test_x, ps_covert_test_x))
            ps_combined_test_y = np.vstack((ps_overt_test_y, ps_covert_test_y))

            print(overt_train_x.shape, covert_train_x.shape, combined_train_x.shape)
            print(ps_overt_train_x.shape, ps_covert_train_x.shape, ps_combined_train_x.shape)

        print("DONE LOADING DATA")

        # full eeg models
        data_pool = run_experiment("art-art", data_pool, overt_train_x, overt_train_y, overt_test_x, overt_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)

        data_pool = run_experiment("art-img", data_pool, overt_train_x, overt_train_y, covert_test_x, covert_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)

        data_pool = run_experiment("img-art", data_pool, covert_train_x, covert_train_y, overt_test_x, overt_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)

        data_pool = run_experiment("img-img", data_pool, covert_train_x, covert_train_y, covert_test_x, covert_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)

        data_pool = run_experiment("combo-combo", data_pool, combined_train_x, combined_train_y, combined_test_x, combined_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)

        data_pool = run_experiment("combo-art", data_pool, combined_train_x, combined_train_y, overt_test_x, overt_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)

        data_pool = run_experiment("combo-img", data_pool, combined_train_x, combined_train_y, covert_test_x, covert_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)


        # prespeech eeg models
        data_pool = run_experiment("ps-art-art", data_pool, ps_overt_train_x, ps_overt_train_y, ps_overt_test_x, ps_overt_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)

        data_pool = run_experiment("ps-art-img", data_pool, ps_overt_train_x, ps_overt_train_y, ps_covert_test_x, ps_covert_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)

        data_pool = run_experiment("ps-img-art", data_pool, ps_covert_train_x, ps_covert_train_y, ps_overt_test_x, ps_overt_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)

        data_pool = run_experiment("ps-img-img", data_pool, ps_covert_train_x, ps_covert_train_y, ps_covert_test_x, ps_covert_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)

        data_pool = run_experiment("ps-combo-combo", data_pool, ps_combined_train_x, ps_combined_train_y, ps_combined_test_x, ps_combined_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)

        data_pool = run_experiment("ps-combo-art", data_pool, ps_combined_train_x, ps_combined_train_y, ps_overt_test_x, ps_overt_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)

        data_pool = run_experiment("ps-combo-img", data_pool, ps_combined_train_x, ps_combined_train_y, ps_covert_test_x, ps_covert_test_y, EPOCHS)
        backup_data_pool(data_pool, filename)