from mne.io import read_epochs_eeglab
from scipy.io import wavfile
from librosa import resample
import numpy as np

from helpers.processing import normalize
from helpers.vad import get_speech_marking_for_file, VAD_SAMPLING_RATE

DATA_LEN = 2048

class DataException(Exception):
    pass


def get_data(subject="sub-01", speech_type="covert", target="ee", epoch=0, eeg_nodes=[], resampleRate=1024.0):
    eeg_data = read_epochs_eeglab(f"./data/derivatives/{subject}/eeg/{subject}_task-{speech_type}-{target}_eeg.set", verbose=False)
    df = eeg_data.to_data_frame()
    epoch_df = df[df["epoch"] == epoch]

    if "FC2" in epoch_df.keys():
        filtered_df = epoch_df.drop(["time", "condition", "epoch", 'FC2'], axis=1)
    else:
        filtered_df = epoch_df.drop(["time", "condition", "epoch"], axis=1)

    # filtered_df = epoch_df[eeg_nodes]

    numpy_df = filtered_df.to_numpy()

    if numpy_df.shape[0] != DATA_LEN:
        raise DataException(f"Invalid data size! Found length of {numpy_df.shape[0]} instead of required {DATA_LEN}")

    # get the audio data
    audio_path = f"./data/sourcedata/{subject}/audio/{subject}_task-overt-{target}_run-{(epoch + 1):02d}_audio.wav"
    rate, audio_data = wavfile.read(audio_path)
    markings = get_speech_marking_for_file(audio_path)



    if resampleRate:
        audio_data =  resample(audio_data / 2**31 , orig_sr=rate, target_sr=resampleRate)
        # create marking array in resampled samplerate
        start_speech = markings[0]['start'] / VAD_SAMPLING_RATE * resampleRate
        start_speech = int(start_speech)
        audio_marking = np.zeros_like(audio_data)
        audio_marking[:start_speech] = 1
  

    else:
        # create resampled marking in audio native rate
        start_speech = markings[0]['start'] / VAD_SAMPLING_RATE * rate
        start_speech = int(start_speech)

        audio_marking = np.zeros_like(audio_data)
        audio_marking[:start_speech] = 1

    audio_data = normalize(audio_data)
    return numpy_df, audio_data, audio_marking