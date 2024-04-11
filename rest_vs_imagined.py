"""
This experiment combines rest data with imagined data to try and confuse the classifier.

If there is not a substantial difference between rest and imagined data, this experiment should show that
"""
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
from tslearn.preprocessing import TimeSeriesScalerMinMax
from keras import backend as K

from sklearn.manifold import TSNE
from tcn import TCN # definition needed for model loading

import random

random.seed(42)



def load_data(speech_type: str, eeg_nodes: list[str], targets: list[str], excluded_subjects=[], num_subjects=20, epochs=20, test_split=0.2, use_filter=False, use_all_nodes=True, use_marking=True):
    xs = []
    ys = []

    test_xs = []
    test_ys = []

    total_data_count = len(targets) * num_subjects
    loaded = 0

    test_array = random.sample(range(epochs), round(test_split * epochs) )
    print(f"Using {test_array} as test")
    
    for i, target in enumerate(targets):

        for j in range(1, num_subjects + 1): # 20 subjects in total
            
            if j in excluded_subjects: # skip excluded subjects
                continue

            subject = f"sub-{j:02}"
            data = read_epochs_eeglab(f"./data/derivatives/{subject}/eeg/{subject}_task-{speech_type}-{target}_eeg.set", verbose=False)
            df = data.to_data_frame()

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
                
                y = [0 for _ in range(len(targets))] # + 1 for rest class
                y[i] = 1 # targets are 0 indexed, rest is always last class

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

def normalize_coeffs(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return mean, std


def generate_tsne_plot(model, directory, filename, xs, ys):
    ["aa", "oo", "ee", "ie", "oe"]
    colors_choices = ["red", "green", "blue", "yellow", "orange", "black"]

    loss, accuracy = model.evaluate(xs, ys)

    mkdir_p(directory + f"/{filename}/" )

    for i, layer in enumerate(model.layers):

        get_dense_out = K.function([model.layers[0].input],
                                [model.layers[i].output])

        result = get_dense_out(xs)[0]

        result_to_use = []
        for r in result:
            result_to_use.append(r.flatten())

        result_to_use = np.array(result_to_use)

        try:
            X_tsne = TSNE(n_components=2, n_jobs=-1, perplexity=3).fit_transform(result_to_use)
        except:
            # skip this layer
            continue
        x = X_tsne[:, 0]
        y = X_tsne[:, 1]
        
        colors = []

        plt.title(f"Layer {i} {layer.name}, Model Accuracy: {accuracy * 100: .2f}")

        for true_y in ys:
            value = np.argmax(true_y)
            colors.append(colors_choices[value])

        sc = plt.scatter(x, y, c=colors)

        plt.savefig(directory + f"/{filename}/" + filename + f"_layer_{i}" + ".png")

        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

def run_experiment(name: str, nodes: list, epochs: int , num_subjects: int, num_repeat: int, use_all_nodes=False):
    """
    Runs 1 set of experiments
    """
    RESULT = []
    TEST_SPLIT = 0.2

    noise = [9, 13, 7, 17, 2]
    # noise = []
    SPEECH_TYPE = "covert"
    TARGETS = ["aa", "oo", "ee", "ie", "oe"]
    # TARGETS = ["ie", "oe"]
    

    print(f"----------------- RUNNING {name} EXPERIMENTS -------------------")
    for i in range(num_repeat):
        try:
            os.remove(f"model_save_{name}_{i}.h5")
        except:
            print("Could not remove file")
            pass

        print(f"RUN {i + 1}/{num_repeat}")
        print("LOADING IMAGINED DATA...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_x, train_y, test_x, test_y = load_data(SPEECH_TYPE, nodes, TARGETS, excluded_subjects=noise, num_subjects=num_subjects, test_split=TEST_SPLIT, use_filter=True, use_marking=False, use_all_nodes=use_all_nodes)

        # print("LOADING REST DATA...")
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     rest_train_x, rest_train_y, rest_test_x, rest_test_y = load_rest_data(nodes, excluded_subjects=noise, num_subjects=num_subjects, test_split=TEST_SPLIT, epochs=20, use_filter=True, use_all_nodes=use_all_nodes, num_targets=len(TARGETS) + 1)

        # #add rest data to train and test data
        # train_x = np.vstack((train_x, rest_train_x))
        # train_y = np.vstack((train_y, rest_train_y))
        # test_x = np.vstack((test_x, rest_test_x))
        # test_y = np.vstack((test_y, rest_test_y))

        # scaler = TimeSeriesScalerMinMax()
        # scaler.fit(np.vstack((train_x, test_x)))

        # train_x = scaler.transform(train_x)
        # test_x = scaler.transform(test_x)
        #minmax scaler per channel
        # for i, _ in enumerate(train_x.T):
        #     scaler = MinMaxScaler()
            
        #     train_channel = train_x[:, :, i]
        #     test_channel = test_x[:, :, i]

        #     total_channel = np.vstack((train_channel, test_channel))

        #     scaler.fit(total_channel)

        #     train_x[:, :, i] = scaler.transform(train_channel)
        #     test_x[:, :, i] = scaler.transform(test_channel)

        # train_x = train_x.reshape((*train_x.shape, 1))
        # test_x = test_x.reshape((*test_x.shape, 1))

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

        model.fit(train_x, train_y, verbose=2, epochs=EPOCHS, shuffle=True, batch_size=32,callbacks=[model_checkpoint_callback], validation_data=(test_x, test_y))

        mkdir_p(f"rest_vs_imagined_result_{SPEECH_TYPE}")
        
        generate_tsne_plot(model, f"rest_vs_imagined_result_{SPEECH_TYPE}", f"_{i}_train_result_{name}", train_x, train_y)

        model = load_model(f"model_save_{name}_{i}.h5",custom_objects={"TCN": TCN})
        generate_tsne_plot(model, f"rest_vs_imagined_result_{SPEECH_TYPE}", f"_{i}_test_result_{name}", test_x, test_y)

        test_loss, test_acc = model.evaluate(test_x, test_y)

        print(test_loss, test_acc)

        results = model.predict(test_x)
        total = 0
        right = 0
        wrong = 0
        
        with open(f"rest_vs_imagined_result_{SPEECH_TYPE}/result_{name}_{i}.txt", "w") as f:
            for j, result in enumerate(results):
                    y_hat = np.argmax(result) 
                    y_real = np.argmax(test_y[j])

                    f.write(f'{y_hat}, {y_real}\n')
                    
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

    EPOCHS = 100
    NUM_SUBJECTS = 20
    
    print("RUNNING REST vs COVERT EXPERIMENT SUITE")

    PAPER_NODES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    POSTER_NODES = ["F7", "F5", "FT7", "FC5", "FC3", "FC1", "T7", "C5", "C3", "Cz", "C4", "TP7", "CP5", "CP3", "P5", "P3"]
    IOANNIS_NODES = ["F3", "F4", "C3", "C4", "P3", "P4"]

    N_REPEATS = 5

    ALL_NODES_RESULT = run_experiment("ALL_NODES", [], epochs=EPOCHS, num_repeat=N_REPEATS, num_subjects=NUM_SUBJECTS, use_all_nodes=True)
    # IOANNIS_NODES_RESULT = run_experiment("IOANNIS_NODES", IOANNIS_NODES, epochs=EPOCHS, num_repeat=N_REPEATS, num_subjects=NUM_SUBJECTS, use_all_nodes=False)
    # POSTER_NODES_RESULT =  run_experiment("POSTER_NODES", POSTER_NODES, epochs=EPOCHS, num_repeat=N_REPEATS, num_subjects=NUM_SUBJECTS, use_all_nodes=False)
    # PAPER_NODES_RESULT =  run_experiment("PAPER_NODES", PAPER_NODES, epochs=EPOCHS, num_repeat=N_REPEATS, num_subjects=NUM_SUBJECTS, use_all_nodes=False)
    
   
    print("------------------------------------------------")
    print()
    print("Baseline", ALL_NODES_RESULT , sum(ALL_NODES_RESULT) / len(ALL_NODES_RESULT))
    # print("Ioannis", IOANNIS_NODES_RESULT , sum(IOANNIS_NODES_RESULT) / len(IOANNIS_NODES_RESULT))
    # print("Poster", POSTER_NODES_RESULT , sum(POSTER_NODES_RESULT) / len(POSTER_NODES_RESULT))
    # print("Paper", PAPER_NODES_RESULT , sum(PAPER_NODES_RESULT) / len(PAPER_NODES_RESULT))