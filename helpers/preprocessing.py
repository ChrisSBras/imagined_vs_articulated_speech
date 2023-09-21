import numpy as np
from librosa.feature import rms

def mark_pre_speech_section(audio_data: np.array) -> np.array:
    audio_data = np.abs(audio_data)
    audio_rms = rms(y=audio_data, hop_length=1, frame_length=8)[0]

    result = []
    end = False

    for i in audio_rms:
        if i > 2.5*  np.mean(audio_rms):
            end = True
        if not end:
            result.append(1)
        else:
            result.append(0)
            
    return np.array(result)


def delete_speech(data: np.array, marking: np.array):
    """
    Removes speech signals from EEG data by using given marking array
    """

    new_data = []

    for channel in data.T:
        new_data.append(channel * marking)

    new_data = np.array(new_data)
    return new_data.T