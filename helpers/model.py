
import tensorflow as tf
import keras.layers

from keras.models import Sequential
from keras import regularizers

def create_save_callback(name, monitor, mode="max"):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{name}.h5",
        save_weights_only=False,
        monitor=monitor,
        mode=mode,
        save_best_only=True)
    
    return model_checkpoint_callback


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

    filter_size = 64
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

    output_layer = keras.layers.Dense(num_y, activation=None)(gap)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    cce = tf.keras.losses.CategoricalCrossentropy()

    model.compile(tf.optimizers.Adam(1e-4), loss='mse', run_eagerly=False,  metrics=["mse"])

    return model 

def create_lstm_model(num_y=5):
    model = Sequential()
    # model.add(LSTM(units=16, input_shape=(2048, 8), return_sequences=False, stateful=False))
    model.add(keras.layers.LSTM(48, activity_regularizer=regularizers.L2(1e-5), return_sequences=True ))
    model.add(keras.layers.Dense(48, activation="tanh", activity_regularizer=regularizers.L2(1e-5)))
    model.add(keras.layers.LSTM(16, activity_regularizer=regularizers.L2(1e-5), return_sequences=False ))
    model.add(keras.layers.Dense(4, activation="relu", activity_regularizer=regularizers.L2(1e-5)))
    model.add(keras.layers.Dense(num_y, activation="softmax"))

    cce = tf.keras.losses.CategoricalCrossentropy()

    model.compile(tf.optimizers.Adam(1e-2), loss=cce, run_eagerly=False,  metrics=["accuracy"])
    
    return model