from scipy.signal import butter, sosfilt
import numpy as np
import matplotlib.pyplot as plt

def filter_data(data: np.array, f1=1, f2=40, fs=1024) -> np.array:

    to_return = []

    for channel in data.T:
        sos = butter(2, [f1, f2], 'bandpass', output='sos', fs=fs)
        filtered_channel = sosfilt(sos, channel)
        to_return.append(filtered_channel)

    to_return = np.array(to_return)
    to_return = to_return.T

    return to_return



