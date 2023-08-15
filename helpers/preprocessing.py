import numpy as np
from librosa.feature import rms

def mark_pre_speech_section(audio_data: np.array) -> np.array:
    audio_rms = rms(y=audio_data, hop_length=1, frame_length=16)[0]

    result = []
    end = False
    for i in audio_rms:
        if i > 0.3:
            end = True
        if not end:
            result.append(1)
        else:
            result.append(0)
            

    return np.array(result)