from mne.io import read_epochs_eeglab
from scipy.io import wavfile
from librosa import resample
import numpy as np

from helpers.processing import normalize
from helpers.vad import get_speech_marking_for_file, VAD_SAMPLING_RATE
from helpers.filtering import filter_data

DATA_LEN = 2048

class DataException(Exception):
    pass


def get_data(subject : str ="sub-01", speech_type : str ="covert", 
             target : str ="ee", epoch : int = 0, eeg_nodes : list=[], 
             resample_rate : float = 1024.0, use_filter : bool =False):
    """
    Reads data from the dataset and returns both audio (if available), speech marking (if audio available)
    and EEG data. 

    Parameters
    -----------
    subject : str
        The subject to get the data from
    
    speech_type : str
        The type of speech to get the data from. Can be either 'overt' or 'covert'

    target : str
        Target sound to get the data from.

    epoch : int
        ID of the epoch to get the data from.
    
    eeg_nodes : list[str]
        List of EEG nodes to use. All nodes not present in this list will be dropped.
        When eeg_nodes is an empty list, no channel dropping is performed.
    
    resample_rate : float
        Sample rate to resample the audio data to
    
    use_filter : bool
        Specifies wether to use a bandpass filter on the EEG signal or not.
    """

    if not speech_type in ["overt", "covert"]:
        raise ValueError(f"{speech_type} is not a valid speech type, please choose either 'overt' or 'covert'")
    
    eeg_data = read_epochs_eeglab(f"./data/derivatives/{subject}/eeg/{subject}_task-{speech_type}-{target}_eeg.set", verbose=False)
    df = eeg_data.to_data_frame()
    epoch_df = df[df["epoch"] == epoch]

    if "FC2" in epoch_df.keys():
        filtered_df = epoch_df.drop(["time", "condition", "epoch", 'FC2'], axis=1)
    else:
        filtered_df = epoch_df.drop(["time", "condition", "epoch"], axis=1)

    if eeg_nodes:
        filtered_df = epoch_df[eeg_nodes]

    numpy_df = filtered_df.to_numpy()
    if use_filter:
        numpy_df = filter_data(numpy_df)

    if numpy_df.shape[0] != DATA_LEN:
        raise DataException(f"Invalid data size! Found length of {numpy_df.shape[0]} instead of required {DATA_LEN}")

    # get the audio data
    audio_path = f"./data/sourcedata/{subject}/audio/{subject}_task-overt-{target}_run-{(epoch + 1):02d}_audio.wav"
    rate, audio_data = wavfile.read(audio_path)
    markings = get_speech_marking_for_file(audio_path)

    if resample_rate:
        audio_data =  resample(audio_data / 2**31 , orig_sr=rate, target_sr=resample_rate)
        # create marking array in resampled samplerate
        audio_marking = np.zeros_like(audio_data)
        if markings:
            start_speech = markings[0]['start'] / VAD_SAMPLING_RATE * resample_rate
            start_speech = int(start_speech)

            audio_marking[:start_speech] = 1
  
    else:
        # create resampled marking in audio native rate
        audio_marking = np.zeros_like(audio_data)
        if markings:
            start_speech = markings[0]['start'] / VAD_SAMPLING_RATE * rate
            start_speech = int(start_speech)


            audio_marking[:start_speech] = 1

    audio_data = normalize(audio_data)
    return numpy_df, audio_data, audio_marking