from mne.io import read_epochs_eeglab
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from librosa import resample
from librosa.feature import rms

from scipy.signal import butter,filtfilt


def moving_average(a, n=200):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]


def get_data(subject="sub-01", speech_type="covert", target="ee", epoch=0, eeg_nodes=[]):
    eeg_data = read_epochs_eeglab(f"./data/derivatives/{subject}/eeg/{subject}_task-{speech_type}-{target}_eeg.set", verbose=False)
    df = eeg_data.to_data_frame()
    epoch_df = df[df["epoch"] == epoch]

    if "FC2" in epoch_df.keys():
        filtered_df = epoch_df.drop(["time", "condition", "epoch", 'FC2'], axis=1)
    else:
        filtered_df = epoch_df.drop(["time", "condition", "epoch"], axis=1)

    filtered_df = epoch_df[eeg_nodes]

    numpy_df = filtered_df.to_numpy()

    # get the audio data
    rate, audio_data = wavfile.read(f"./data/sourcedata/{subject}/audio/{subject}_task-overt-{target}_run-{(epoch + 1):02d}_audio.wav")
    
    audio_data =  resample(audio_data / 2**31 , orig_sr=rate, target_sr=1024.0)

    return numpy_df, audio_data

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





if __name__ == "__main__":
    
    EEG_NODES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

    fs = 1024.0
    cutoff_low = 8
    cutoff_high = 12.0

    TARGET = "aa"


    for epoch in range(0, 20):
        eeg_data_covert, audio_data = get_data(subject="sub-05", speech_type="covert", epoch=epoch, eeg_nodes=EEG_NODES,target=TARGET)
        eeg_data_overt, audio_data = get_data(subject="sub-05", speech_type="overt", epoch=epoch, eeg_nodes=EEG_NODES, target=TARGET)


        fig, (ax1, ax2, ax3) = plt.subplots(3)

        fig.set_size_inches(16, 10)
        
        for i, data in enumerate(eeg_data_covert.T):
            # ax1.plot(band_filter(i, cutoff_low, cutoff_high, fs, 2))
            # ax1.plot(i)
            ax1.plot(moving_average(data))
            ax2.plot(moving_average(eeg_data_overt.T[i]))

        ax1.legend(EEG_NODES)
        ax2.legend(EEG_NODES)

        ax1.set_title("Covert")
        ax2.set_title("Overt")
        
        ax3.plot(audio_data)
        audio_rms = rms(y=audio_data, hop_length=1, frame_length=16)[0]
        ax3.plot(audio_rms)
        print(audio_rms)
        ax3.legend(["Waveform", "RMS"])

        plt.show()

        exit()