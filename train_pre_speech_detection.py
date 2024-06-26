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

from helpers.filtering import filter_data, rereference, normalize_df
from helpers.preprocessing import delete_speech
from helpers.model import create_conv_model_regression, create_wavenet_model
from helpers.vad import get_speech_marking_for_file, VAD_SAMPLING_RATE

import os

import random

random.seed(43)

def load_data(speech_type: str, eeg_nodes: list[str], targets: list[str], excluded_subjects=[], num_subjects=20, epochs=20, test_split=0.2, use_filter=True, use_all_nodes=False, use_marking=True):
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
                    continue
                # numpy_df = normalize_df(numpy_df)
           
                # get audio data
                if speech_type == "overt" and use_marking:
                    audio_path = f"./data/sourcedata/{subject}/audio/{subject}_task-overt-{target}_run-{(epoch + 1):02d}_audio.wav"
                    markings = get_speech_marking_for_file(audio_path)

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



if __name__ == "__main__":
    try:
        os.remove("model_save_prespeech_detection.h5")
    except:
        pass

    try:
        import sys
        channel_set = int(sys.argv[1])
    except:
        print("No channelset selected")
        exit(1)
    

    TARGETS = ["aa", "oo", "ee", "ie", "oe"]
    # TARGETS = ["aa", "oo", "ee"]
    # TARGETS = ["aa"]
    EEG_NODES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    EEG_NODES2 = ["F7", "F5", "FT7", "FC5", "FC3", "FC1", "T7", "C5", "C3", "Cz", "C4", "TP7", "CP5", "CP3", "P5", "P3"]
    EEG_NODES_IOANNIS = ["F3", "F4", "C3", "C4", "P3", "P4"]
    SPEECH_TYPE = "overt"

    channel_sets = [EEG_NODES_IOANNIS, EEG_NODES, EEG_NODES2]

    N_REPEATS = 5
    N_SUBJECTS = 20
    EPOCHS = 150

    noise = [9, 13, 7, 17, 2]

    BASE_LINE_RESULTS = []
    TEST_RESULT = []

    RESULT_MSE = []
    RESULT_MAE = []

    use_all_nodes = False
    try:
        channeL_set_to_use = channel_sets[channel_set]
    except:
        channeL_set_to_use = []
        use_all_nodes = True

    for run_id in range(N_REPEATS):
        try:
            os.remove("model_save_prespeech_detection.h5")
        except:
            pass
        print("LOADING DATA...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_x, train_y, test_x, test_y = load_data(SPEECH_TYPE, channeL_set_to_use, TARGETS, excluded_subjects=noise, num_subjects=N_SUBJECTS, test_split=0.2, use_filter=True, use_marking=True, use_all_nodes=use_all_nodes)

        print(train_y)

        print("DONE LOADING DATA")
        # create a new model fo this run
        model = create_conv_model_regression(input_shape=(2048, train_x.shape[2]), num_y=train_y.shape[-1])
        model.summary()

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"model_save_prespeech_detection.h5",
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        model.fit(train_x, train_y, verbose=2, epochs=EPOCHS, shuffle=True, batch_size=16,callbacks=[model_checkpoint_callback], validation_data=(test_x, test_y))
        model.save("model.h5")
        model.save(f"./prespeech_final_results/models/model_{channel_set}_run_{run_id}.h5")

        model = load_model(f"model_save_prespeech_detection.h5")
        model.save(f"./prespeech_final_results/models/model_{channel_set}_run_{run_id}.h5")

        loss, mse, mae = model.evaluate(test_x, test_y)

        RESULT_MSE.append(mse)
        RESULT_MAE.append(mae)

        print(loss, mse, mae)

        results = model.predict(test_x)

        # for i, y_hat in enumerate(results):
        #     y_hat = int(y_hat * 2048)
        #     real = int(test_y[i] * 2048)

        #     plt.plot(test_x[i])
        #     plt.axvline(x = real, color = 'b', label = 'Real')
        #     plt.axvline(x = y_hat, color = 'r', label = 'Predicted')

        #     plt.legend()
        #     plt.savefig(f"prespeech_vis/run{run_id}_result{i}")


    with open(f"./prespeech_final_results/result_{channel_set}", "w") as f:
        print("------------------------------------------------")
        print()
        print("Baseline MSE", RESULT_MSE , sum(RESULT_MSE) / len(RESULT_MSE))
        print("Baseline MAE", RESULT_MAE , sum(RESULT_MAE) / len(RESULT_MAE))

        for i in channeL_set_to_use:
            f.write(f"{i}, ")
        f.write("\n")
        f.write(f"Baseline MAE, {RESULT_MAE}, {sum(RESULT_MAE) / len(RESULT_MAE)}\n")
        f.write(f"Baseline MSE, {RESULT_MSE}, {sum(RESULT_MSE) / len(RESULT_MSE)}\n")
        f.close()