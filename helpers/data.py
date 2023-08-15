from mne.io import read_epochs_eeglab
from scipy.io import wavfile
from librosa import resample

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