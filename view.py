
import matplotlib.pyplot as plt

from helpers.processing import moving_average, normalize
from helpers.preprocessing import mark_pre_speech_section
from helpers.data import get_data
from helpers.filtering import filter_data
from helpers.preprocessing import delete_speech

if __name__ == "__main__":
    
    EEG_NODES = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

    fs = 1024.0
    cutoff_low = 8
    cutoff_high = 12.0

    TARGET = "aa"

    for epoch in range(0, 20):
        eeg_data_covert, audio_data, prespeech_marking = get_data(subject="sub-01", speech_type="covert", epoch=epoch, eeg_nodes=EEG_NODES,target=TARGET, use_filter=True )
        eeg_data_overt, audio_data, prespeech_marking = get_data(subject="sub-01", speech_type="overt", epoch=epoch, eeg_nodes=EEG_NODES, target=TARGET, use_filter=True)
        audio_data = normalize(audio_data)

        fig, (ax1, ax2, ax3) = plt.subplots(3)

        fig.set_size_inches(16, 10)
        
        for i, data in enumerate(eeg_data_covert.T):
            # ax1.plot(band_filter(i, cutoff_low, cutoff_high, fs, 2))
            # ax1.plot(i

            data_covert = data
            data_overt = eeg_data_overt.T[i]

            ax1.plot(data_covert)
            ax2.plot(data_overt)

        ax1.legend(EEG_NODES)
        ax2.legend(EEG_NODES)

        ax1.set_title("Covert")
        ax2.set_title("Overt")
        
        ax3.plot(audio_data)

        ax3.plot(prespeech_marking)

        ax3.legend(["Waveform", "Marker"])

        plt.show()

        # exit()