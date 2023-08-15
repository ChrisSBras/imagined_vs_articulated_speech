import numpy as np
from librosa.feature import rms

def mark_pre_speech_section(audio_data: np.array) -> np.array:
    audio_data = np.abs(audio_data)
    audio_rms = rms(y=audio_data, hop_length=1, frame_length=8)[0]
    audio_rms2 = rms(y=audio_data, hop_length=1, frame_length=audio_data.shape[0])[0]
    result = []
    end = False

    print("MEAN", np.mean(audio_rms), audio_rms2)

    for i in audio_rms:
        if i > 2.5*  np.mean(audio_rms):
            end = True
        if not end:
            result.append(1)
        else:
            result.append(0)
            
    return np.array(result)