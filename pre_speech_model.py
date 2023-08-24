import warnings
import random
random.seed(69)

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Conv1D
from tensorflow.keras.utils import Sequence
import tensorflow as tf

HIDDEN_SIZE = 56

import keras.backend as K

from helpers.preprocessing import mark_pre_speech_section
from helpers.data import get_data, DataException
from helpers.model import create_save_callback
from helpers.processing import moving_average, butter_lowpass_filter


TIME_STEPS = 128

def generate_data_line(xs, ys, length=2048, window=1, features=1):
    x_out = []
    y_out = []
    for i in range(0, length * window - TIME_STEPS, window):
        current_x = np.asarray( xs[i: i+TIME_STEPS].reshape((-features, features))).astype('float32')
        current_y = np.asarray([ys[i + TIME_STEPS]]).astype('float32')
        x_out.append(current_x)
        y_out.append(current_y)

    return np.asarray(x_out), np.asarray(y_out).astype('float32')

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
            test_array = random.sample(range(epochs), round(test_split * epochs) )
            print(f"Using epochs as test for subject {subject} target {target}: ", test_array)

            for epoch in range(epochs):
                try:
                    eeg_data, audio_data, audio_marking = get_data(subject=subject, speech_type=speech_type, target=target, eeg_nodes=eeg_nodes, epoch=epoch)
                except DataException as e:
                    print(e)
                    continue

                x_data, y_data = generate_data_line(eeg_data, audio_marking, features=eeg_data.shape[1])

                if epoch in test_array:
                    test_xs.append(x_data)
                    test_ys.append(y_data)
                else:
                    xs.append(x_data)
                    ys.append(y_data)

    return np.array(xs), np.array(ys), np.array(test_xs), np.array(test_ys)


def generate_model(features=16, batch_size=16):
    model = Sequential()
    # model.add(Conv1D(1, 128, activation=None))
    model.add(LSTM(units=HIDDEN_SIZE, batch_input_shape=(batch_size, TIME_STEPS, features), return_sequences=False, stateful=False))
    # model.add(Conv1D(1, 128, activation=None))
    # model.add(LSTM(units=HIDDEN_SIZE // 2,  return_sequences=False))
    
    # model.add(LSTM(units=HIDDEN_SIZE))
    model.add(Dense(1, activation=None))

    cce = tf.keras.losses.CategoricalCrossentropy()
    model.compile(tf.optimizers.Adam(), loss='mse', run_eagerly=False)
    return model


def visual_validation(model, speech_type: str, eeg_nodes: list[str], targets: list[str], excluded_subjects=[], num_subjects=20, epochs=20, test_split=0.2):
    for i, target in enumerate(targets):

        for j in range(1, num_subjects + 1): # 20 subjects in total
            
            if j in excluded_subjects: # skip excluded subjects
                continue

            subject = f"sub-{j:02}"
            test_array = random.sample(range(epochs), round(test_split * epochs) )
            print(f"Using epochs as test for subject {subject} target {target}: ", test_array)

            for epoch in range(epochs):
                if epoch in test_array:
                    eeg_data, audio_data, audio_marking = get_data(subject=subject, speech_type=speech_type, target=target, eeg_nodes=eeg_nodes, epoch=epoch)

                    zeros_eeg = np.zeros((TIME_STEPS, eeg_data.shape[1]))
                    zeros_marking =  np.zeros((TIME_STEPS))

                    x_data, _ = generate_data_line(np.vstack((zeros_eeg, eeg_data)), np.hstack((zeros_marking, audio_marking)), features=eeg_data.shape[1])
                    y_hat = model.predict(x_data, verbose=1, batch_size=BATCH_SIZE)

                    output_signal_normal = np.array([y[0] for y in y_hat])
                    
                    # post process model output
                    # output_signal_normal = butter_lowpass_filter(output_signal_normal, 2, 1024, 4) * 2
                    # output_signal_normal = np.clip(output_signal_normal, 0, 1)


                    fig, (ax1, ax2) = plt.subplots(2)
                    fig.set_size_inches(16, 10)

                    ax1.plot(audio_data)
                    ax1.plot(audio_marking)
                    ax1.plot(output_signal_normal)

                    ax1.plot()
                    ax1.legend(["Audio", "True Marking", "Predicted Marking"])
                    ax1.set_title(f"{subject} {target} {epoch} Audio")

                    eeg_max = np.max(np.abs(eeg_data.T))
                    for i, data in enumerate(eeg_data.T):
                        val = np.max(moving_average(data))
                        if val > eeg_max:
                            eeg_max = val

                    ax2.plot(audio_marking * eeg_max)
                    ax2.plot(output_signal_normal * eeg_max)
                    for i, data in enumerate(eeg_data.T):
                        ax2.plot(moving_average(data))

                    ax2.plot()
                    ax2.legend(["True Marking", "Predicted Marking"])
                    ax2.set_title(f"{subject} {target} {epoch} EEG")

                    plt.show()

                else:
                    continue

if __name__ == "__main__":

    BATCH_SIZE = 16
    EPOCHS= 20

    TARGETS = ["aa", "oo", "ee", "ie", "oe"]
    EEG_NODES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    EEG_NODES2 = ["F7", "F5", "FT7", "FC5", "FC3", "FC1", "T7", "C5", "C3", "Cz", "C4", "TP7", "CP5", "CP3", "P5", "P3"]
    SPEECH_TYPE = "overt"

    females = [1, 3, 4, 5, 6, 7, 8, 9, 10, 14, 16, 17, 18, 19]
    noise = [9, 13, 7, 17, 2]

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     train_x, train_y, test_x, test_y = load_data(SPEECH_TYPE, EEG_NODES2, TARGETS, excluded_subjects=noise, num_subjects=1, test_split=0.2)

    # np.save("train_x_prespeech", train_x)
    # np.save("train_y_prespeech", train_y)
    # np.save("test_x_prespeech", test_x)
    # np.save("test_y_prespeech", test_y)

    train_x = np.load("train_x_prespeech.npy")
    train_y = np.load("train_y_prespeech.npy")

    test_x = np.load("test_x_prespeech.npy")
    test_y = np.load("test_y_prespeech.npy")

   
    train_x_flat = []
    train_y_flat = []
    for i in train_x:
        train_x_flat.extend(i)

    for i in train_y:
        train_y_flat.extend(i)
    
    train_x_flat = np.array(train_x_flat)
    train_y_flat = np.array(train_y_flat)
    print(train_x.shape, train_y.shape)
    print(train_x_flat.shape, train_y_flat.shape)

    features = train_x.shape[-1]
    model = generate_model(features=features, batch_size=BATCH_SIZE)
    model.summary()

    model = load_model("prespeech_save_all_nodes.h5")
    # save_callback = create_save_callback("prespeech_save_all_nodes", "loss", "min")
    # model.fit(train_x_flat[:30000], train_y_flat[:30000], epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE, callbacks=[save_callback])

    # model = load_model("prespeech_save.h5")

    model.save("prespeech_all_nodes.h5")

    test_x_flat = []
    test_y_flat = []
    for i in test_x:
        test_x_flat.extend(i)

    for i in train_y:
        test_y_flat.extend(i)

    test_x_flat = np.array(test_x_flat)
    test_y_flat = np.array(test_y_flat)

    visual_validation(model, SPEECH_TYPE, EEG_NODES2, TARGETS, excluded_subjects=noise, num_subjects=1, test_split=0.2)

    y_hat = model.predict(test_x_flat[:2048 * 10], verbose=1, batch_size=BATCH_SIZE)
    target_signal = test_y_flat[:2048 * 10]

    output_signal_normal = np.array([y[0] for y in y_hat])

    #cleanup signal
   
    output_signal_normal = butter_lowpass_filter(output_signal_normal, 2, 1024, 4) * 2
    output_signal_normal = np.clip(output_signal_normal, 0, 1)

    # output_signal_normal[np.where(output_signal_normal >= 0.5)] = 1
    # output_signal_normal[np.where(output_signal_normal < 0.5)] = 0

    fig, (ax1) = plt.subplots(1)

    fig.set_size_inches(16, 10)

    ax1.plot(output_signal_normal)
    ax1.plot(target_signal)
    ax1.legend(["Prediction", "Target"])
    plt.show()