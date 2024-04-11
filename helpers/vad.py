VAD_SAMPLING_RATE = 16000

# import torch
# # torch.set_num_threads(16)
# try:
#     model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
#                                 model='silero_vad',
#                                 force_reload=True,
#                                 onnx=False)

#     (get_speech_timestamps,
#     save_audio,
#     read_audio,
#     VADIterator,
#     collect_chunks) = utils
# except:
#     model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
#                                 model='silero_vad',
#                                 force_reload=False,
#                                 onnx=False)

#     (get_speech_timestamps,
#     save_audio,
#     read_audio,
#     VADIterator,
#     collect_chunks) = utils

def get_speech_marking_for_file(filepath: str) -> dict:
    """
    Returns the beginning and end of speech for a given wave file
    """
    wav = read_audio(filepath, sampling_rate=VAD_SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=VAD_SAMPLING_RATE)
    return speech_timestamps


# if __name__ == "__main__":
#     PATH = "../data/sourcedata/sub-01/audio/sub-01_task-overt-aa_run-01_audio.wav"

#     timestamps = get_speech_marking_for_file(PATH)
#     print(timestamps)