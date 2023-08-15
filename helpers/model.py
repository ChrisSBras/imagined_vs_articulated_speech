
import tensorflow as tf
def create_save_callback(name, monitor, mode="max"):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{name}.h5",
        save_weights_only=False,
        monitor=monitor,
        mode=mode,
        save_best_only=True)
    
    return model_checkpoint_callback