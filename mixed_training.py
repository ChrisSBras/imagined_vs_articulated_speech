"""
This script runs the mixed training experiments. Mixed training in this case
means that the model for classification is trained on both overt and covert data, but only validated
with one of the two.
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
from helpers.analysis import count_correct_predictions

import random

random.seed(42)

def load_data(speech_type: str, eeg_nodes: list[str], targets: list[str], excluded_subjects=[], num_subjects=20, epochs=20, test_split=0.2, use_filter=False, use_all_nodes=True, use_marking=True, invert_marking=False):
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

                        if invert_marking:
                            audio_marking[start_speech:] = 1 # inverted selection now
                        else:
                            audio_marking[:start_speech] = 1

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

def run_experiment(name: str, nodes: list, epochs: int , num_subjects: int, num_repeat: int, use_all_nodes=True, use_marking=True, invert_marking=False):
    """
    Runs 1 set of experiments
    """
    OVERT_RESULT = []
    COVERT_RESULT = []

    noise = [9, 13, 7, 17, 2]
    SPEECH_TYPE = "overt"
    TARGETS = ["aa", "oo", "ee", "ie", "oe"]

    print(f"----------------- RUNNING {name} EXPERIMENTS -------------------")
    for i in range(num_repeat):
        try:
            os.remove(f"cover_overt_classification_model_save_{name}_{i}.h5")
        except:
            pass

        print(f"RUN {i + 1}/{num_repeat}")
        print("LOADING OVERT DATA...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            overt_train_x, overt_train_y, overt_test_x, overt_test_y = load_data("overt", nodes, TARGETS, excluded_subjects=noise, num_subjects=num_subjects, test_split=0.15, use_filter=True, use_marking=use_marking, use_all_nodes=use_all_nodes, invert_marking=invert_marking)

        print("LOADING COVERT DATA...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            covert_train_x, covert_train_y, covert_test_x, covert_test_y = load_data("covert", nodes, TARGETS, excluded_subjects=noise, num_subjects=num_subjects, test_split=0.15, use_filter=True, use_marking=False, use_all_nodes=use_all_nodes)

        print("DONE LOADING DATA")

        total_train_x = np.concatenate((covert_train_x, overt_train_x))
        total_train_y =  np.concatenate((covert_train_y, overt_train_y))

        total_test_x = np.concatenate((covert_test_x, overt_test_x))
        total_test_y = np.concatenate((covert_test_y, overt_test_y))

        # create a new model fo this run
        model = create_conv_model(input_shape=(total_train_x.shape[1], overt_train_x.shape[2]), num_y=overt_train_y.shape[-1])
        model.summary()

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"cover_overt_classification_model_save_{name}_{i}.h5",
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        model.fit(total_train_x, total_train_y, verbose=2, epochs=EPOCHS, shuffle=True, batch_size=64,callbacks=[model_checkpoint_callback], validation_data=(total_test_x, total_test_y))

        model = load_model(f"cover_overt_classification_model_save_{name}_{i}.h5")

        test_loss, test_acc = model.evaluate(total_test_x, total_test_y)

        print(test_loss, test_acc)

        overt_results = model.predict(overt_test_x)
        covert_results = model.predict(covert_test_x) # no covert training

        overt_correct, overt_total = count_correct_predictions(overt_results, overt_test_y)
        covert_correct, covert_total = count_correct_predictions(covert_results, covert_test_y)

        print(f"predicted {overt_correct} / {overt_total} correct, accuracy of {(overt_correct / overt_total) * 100:.2f}% for OVERT classification")
        print(f"predicted {covert_correct} / {covert_total} correct, accuracy of {(covert_correct / covert_total) * 100:.2f}% for COVERT classification")

        OVERT_RESULT.append(overt_correct / overt_total)
        COVERT_RESULT.append(covert_correct / covert_total)

    print(f"--------------- {name} RESULT DONE ---------------")
    print(f"OVERT Average accuracy: {sum(OVERT_RESULT)/len(OVERT_RESULT)* 100:.2f}%")
    print(f"COVERT Average accuracy: {sum(COVERT_RESULT)/len(COVERT_RESULT)* 100:.2f}%")

    return OVERT_RESULT, COVERT_RESULT

if __name__ == "__main__":
    EPOCHS = 200
    NUM_SUBJECTS = 20
    N_REPEATS = 3
    
    print("RUNNING OVERT MODEL COVERT CLASSIFICATION EXPERIMENT SUITE")
    result_overt, result_covert= run_experiment("MIXED TRAINING", [], epochs=EPOCHS, num_repeat=N_REPEATS, num_subjects=NUM_SUBJECTS, use_all_nodes=True, use_marking=False)

    print("------------------------------------------------")
    print()
    print("OVERT", result_overt , sum(result_overt) / len(result_overt))
    print("COVERT", result_covert , sum(result_covert) / len(result_covert))
