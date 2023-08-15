import numpy as np
from scipy.signal import butter,filtfilt

def moving_average(a, n=200):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = fs / 2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order):
    nyq = fs / 2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def band_filter(data, cutoff_low, cutoff_high, fs, order):
    data = butter_lowpass_filter(data, cutoff_high, fs, order)
    data = butter_highpass_filter(data, cutoff_low, fs, order)
    return data

def normalize(x: np.array) -> np.array:
    return (2 * (x-np.min(x))/(np.max(x)-np.min(x))) - 1.0