## short term memory

### 1. Goal

The goal of the project is predicting whether a patient has a short-term memory disturbance, based on EEG data.

The idea is to 

- transform the raw EEG signal by Fourier or wavelet transform 

- and then input it into a classifier, e.g. an xgboost, convolutional or recurrent neural network.

*So far, none of these methods brought a satisfactory predictive model.*

The project is based on the [paper published by Yuri Pavlov, Boris Kotchoubey](https://www.researchgate.net/publication/344430052_The_electrophysiological_underpinnings_of_variation_in_verbal_working_memory_capacity).


### 2. Run the code

In order to run the analysis, you need to:

- download this repo

- install dependecies from `requirements.txt`, e.g. with `pip install -r requirements.txt`

- download the publicly available raw data with ```aws s3 sync --no-sign-request s3://openneuro.org/ds003655 ds003655-download/```

- update the `DATA_RAW_PATH` in the `config.py` accordingly. You may also want to update the other parameters in `config.py`

- run the `api.py` script: it will transform the raw data (which was saved by matlab - eeglab) into a more Python-frendly, convenient and efficient format

- run the scripts starting with `run...`: `xgb`, `wavelet_cnn` and `fourier_cnn` - they will train these models and print basic validation metrics. You can adjust these files anyway you want, by changing hyperparameters, transformers etc.


### 3. TODO

- !tests!

- plotting wavelet transform, just as in the paper

- joining phases: encoding and delay

- explorartory data analysis, in partcular: differences in wavelengths between persons
