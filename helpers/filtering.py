from scipy.signal import butter, sosfilt
import numpy as np
import matplotlib.pyplot as plt

def filter_data(data: np.array, f1=0.1, f2=40, fs=1024) -> np.array:
    """
    Applies bandpass filter on data between given frequencies f1 and f2 with samplerate fs.
    """
    to_return = []

    for channel in data.T:
        sos = butter(2, [f1, f2], 'bandpass', output='sos', fs=fs)
        filtered_channel = sosfilt(sos, channel)
        to_return.append(filtered_channel)

    to_return = np.array(to_return)
    to_return = to_return.T

    return to_return


def rereference(data: np.array) -> np.array:
    """
    Rereference signal by taking average of all channels and subtracting
    that from all channels
    """

    average = np.average(data, axis=1)

    new_data = []

    for channel in data.T:
        new_data.append(channel - average)

    new_data = np.array(new_data)
    return new_data.T