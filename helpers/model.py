
import tensorflow as tf
import keras.layers

from keras.models import Sequential
from keras import regularizers

from tcn import TCN
from tcn import compiled_tcn
from tslearn.metrics import dtw_path
from collections import defaultdict
import numpy as np


def create_save_callback(name, monitor, mode="max"):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{name}.h5",
        save_weights_only=False,
        monitor=monitor,
        mode=mode,
        save_best_only=True)
    
    return model_checkpoint_callback



def create_conv_layer(layer, filters, kernel_size):
    to_return = keras.layers.Conv1D(
        filters=filters, 
        kernel_size=kernel_size, 
        # kernel_initializer="random_normal",
        # bias_initializer="random_normal"
    )(layer)

    to_return = keras.layers.ReLU()(to_return)
    to_return = keras.layers.BatchNormalization()(to_return)

    # to_return = keras.layers.Dropout(0.9)(to_return)
    return to_return

def create_wavenet_model(input_shape=(2048, 61, 1), num_y=5):
    dilations = (1, 4, 16, 64, 256, 1024)

    dilations = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    input_layer = keras.layers.Input(input_shape)
    #trying wavenet style model
    tcn_layer = TCN(
        nb_filters=20,
        kernel_size=6,
        use_batch_norm=True, 
        use_layer_norm=False,
        use_skip_connections=True,
        padding="causal",
        dropout_rate= 0.2,
        dilations=dilations
    )(input_layer)

    output_layer = keras.layers.Dense(num_y, activation="softmax")(tcn_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    cce = tf.keras.losses.CategoricalCrossentropy()

    model.compile(tf.optimizers.Adam(1e-2), loss=cce, run_eagerly=False,  metrics=["accuracy"])

    # model.compile(tf.optimizers.Adam(1e-2), loss="mse", run_eagerly=False,  metrics=["mse", "mae"])

    return model 

# def create_conv_model(input_shape=(2048, 61, 1), num_y=5):


#     dilations = (1, 4, 16, 64, 256, 1024)

#     dilations = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
#     input_layer = keras.layers.Input(input_shape)
#     #trying wavenet style model
    

#     tcn_layer = TCN(
#         nb_filters=20,
#         kernel_size=6,
#         use_batch_norm=False, 
#         use_layer_norm=False,
#         use_skip_connections=True,
#         padding="causal",
#         dropout_rate= 0,
#         dilations=dilations
#     )(input_layer)


#     filter_size = 512
#     kernel_size = 3
    
#     # fc = keras.layers.Dense(256, activation="relu")(input_layer)
#     conv = create_conv_layer(input_layer, 256, 6)
#     conv = create_conv_layer(conv, 128, 6)
#     conv = create_conv_layer(conv, 64, 6)
#     conv = create_conv_layer(conv, 32, 6)
#     # conv = create_conv_layer(conv, 5, 128)

#     # gap = keras.layers.GlobalAveragePooling1D()(tcn_layer)
#     fc = keras.layers.Dense(32, activation="relu")(tcn_layer)
#     # fc = keras.layers.Dropout(0.1)(fc)
#     fc = keras.layers.Dense(64, activation="relu")(tcn_layer)
#     # fc = keras.layers.Dense(32, activation="relu")(fc)
#     # fc = keras.layers.Dense(32, activation="relu")(fc)

#     output_layer = keras.layers.Dense(num_y, activation="tanh")(tcn_layer)

#     model = keras.models.Model(inputs=input_layer, outputs=output_layer)

#     cce = tf.keras.losses.CategoricalCrossentropy()

#     model.compile(tf.optimizers.Adam(1e-2), loss=cce, run_eagerly=False,  metrics=["categorical_accuracy"])

#     return model 

def create_conv_model(input_shape=(2048, 61), num_y=5):
    input_layer = keras.layers.Input(input_shape)

    filter_size = 32
    kernel_size = 3

    conv1 = keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv1 = keras.layers.Dropout(0.25)(conv1)

    conv2 = keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv2 = keras.layers.Dropout(0.25)(conv2)

    conv3 = keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    conv3 = keras.layers.Dropout(0.25)(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_y, activation="softmax")(gap)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    cce = tf.keras.losses.CategoricalCrossentropy()

    model.compile(tf.optimizers.Adam(1e-2), loss=cce, run_eagerly=False,  metrics=["accuracy"])

    return model 


def create_conv_model_regression(input_shape=(2048, 61), num_y=1):
    input_layer = keras.layers.Input(input_shape)

    filter_size = 3
    kernel_size = 2048

    conv1 = keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv1 = keras.layers.Dropout(0.25)(conv1)

    conv2 = keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv2 = keras.layers.Dropout(0.25)(conv2)

    conv3 = keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    conv3 = keras.layers.Dropout(0.25)(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_y, activation=None)(gap)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    cce = tf.keras.losses.CategoricalCrossentropy()

    model.compile(tf.optimizers.Adam(1e-4), loss='mse', run_eagerly=False,  metrics=["mse", "mae"])

    return model 

def create_lstm_model(input_shape=(2048, 61), num_y=5):
    model = Sequential()
    model.add(keras.layers.LSTM(units=16, input_shape=input_shape, return_sequences=False, stateful=False, dropout=0,
    recurrent_dropout=0,))
    # model.add(keras.layers.LSTM(48, activity_regularizer=regularizers.L2(1e-5), return_sequences=True ))
    # model.add(keras.layers.Dense(48, activation="tanh", activity_regularizer=regularizers.L2(1e-5)))
    # model.add(keras.layers.LSTM(16, activity_regularizer=regularizers.L2(1e-5), return_sequences=False ))
    # model.add(keras.layers.Dense(4, activation="relu", activity_regularizer=regularizers.L2(1e-5)))
    model.add(keras.layers.Dense(num_y, activation="softmax"))

    cce = tf.keras.losses.CategoricalCrossentropy()

    model.compile(tf.optimizers.Adam(1e-2), loss=cce, run_eagerly=False,  metrics=["accuracy"])
    
    return model

def create_lstm_model(input_shape=(2048, 61), num_y=5):
    model = Sequential()
    model.add(keras.layers.LSTM(units=16, input_shape=input_shape, return_sequences=False, stateful=False, dropout=0,
    recurrent_dropout=0,))
    # model.add(keras.layers.LSTM(48, activity_regularizer=regularizers.L2(1e-5), return_sequences=True ))
    # model.add(keras.layers.Dense(48, activation="tanh", activity_regularizer=regularizers.L2(1e-5)))
    # model.add(keras.layers.LSTM(16, activity_regularizer=regularizers.L2(1e-5), return_sequences=False ))
    # model.add(keras.layers.Dense(4, activation="relu", activity_regularizer=regularizers.L2(1e-5)))
    model.add(keras.layers.Dense(num_y, activation="softmax"))

    cce = tf.keras.losses.CategoricalCrossentropy()

    model.compile(tf.optimizers.Adam(1e-2), loss=cce, run_eagerly=False,  metrics=["accuracy"])
    
    return model

def make_timeseries_knn_prediction(train_x, train_y, sample, n=7):
    distances = []

    for i, val in enumerate(train_x):

        label = train_y[i]

        _, distance = dtw_path(val, sample)

        distances.append((label, distance))

    distances.sort(key=lambda x: x[1])

    result = [0 for i in range(train_y.shape[1])]

    for i in range(n):
        label = np.argmax(distances[i][0])
        result[label] += 1

    result = np.array(result)

    label_result = [0 for i in range(train_y.shape[1])]
    label_result[np.argmax(result)] = 1
    label_result = np.array(label_result)

    return distances[0][0]
