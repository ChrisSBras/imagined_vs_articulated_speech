# Speech Classification Using EEG Data

## Installation
To run the models, first you need to install the required python packages.

```bash
pip install -r requirements.txt
```

## Running the Overt Speech Model
To run the baseline overt speech model, simply run the script `run_model.py`. This will first generate the data and save it for further use. 

Parameters for the run can currently only be changed in the script itself. A more elegant way to handle this is on its way.

```bash
python run_model.py
```

## Viewer Module
To get better insights into what the signals look like, a viewer model is included. This can be run by running `view.py`. It will read the data from each subject and display it together with audio and speech marking.

```bash
python view.py
```

## Prespeech Module
This module is currently being worked on and may not run as expected.
```bash
python pre_speech_model.py
```