# %%
# https://openneuro.org/datasets/ds003655/versions/1.0.0
from pathlib import Path
path = Path("/home/tomek/doktorat/ds003655-download/sub-001/eeg")
import mne
import pandas as pd
import numpy as np

channels = pd.read_csv(path / "sub-001_task-VerbalWorkingMemory_channels.tsv", sep="\t")
eog = channels.loc[channels.type == "EOG", "name"].tolist()
raw_data = mne.io.read_raw_eeglab(input_fname=path / "sub-001_task-VerbalWorkingMemory_eeg.set", eog=eog, preload=True)
raw_data.apply_function(lambda x: x / 1000000)

raw_data.plot_psd(fmax=50)
raw_data.plot(start=0, n_channels=1)

data = raw_data.get_data(units='uV')

len(raw_data)
len(raw_data[0])
raw_data[0][0].shape
raw_data[0][1].shape

raw_data[1]
raw_data[1][1].shape

# %%

events = pd.read_csv(path / "sub-001_task-VerbalWorkingMemory_events.tsv", sep="\t")
events['trial_type'].value_counts()
events.shape
raw_data
events['trial_type'].tolist()[:100]
events[['onset', 'trial_type']][:100]
events['onset'].tolist()
events[events['trial_type'].apply(lambda x: x.startswith("start of the baseline period"))]['trial_type'].value_counts()
events[events['trial_type'].apply(lambda x: x.startswith("Response"))]['trial_type'].value_counts()

difficulties = []
all_events = []
event_path = "/home/tomek/doktorat/ds003655-download/sub-{num}/eeg/sub-{num}_task-VerbalWorkingMemory_events.tsv"
for i in range(1, 157):
    num = str(i)
    if i < 10:
        num = "0" + num
    if i < 100:
        num = "0" + num
    events = pd.read_csv(event_path.format(num=num), sep="\t")
    all_events.append(events)
    responses = events[events['trial_type'].apply(lambda x: x.startswith("Response: 5R"))]
    corrects = np.array(responses['trial_type'] == "Response: 5R: correct")
    difficulties.append(corrects)

all_events

all_events[0]['trial_type'].value_counts()
all_events[0][all_events[0]['trial_type'].apply(lambda x: x.startswith("Response: 5R"))]
[(i, y[y['trial_type'].apply(lambda x: x.startswith("Response: 5R"))].shape) for (i, y) in enumerate(all_events)]

all_events[118]

raw_data.units
raw_data.plot_psd(fmax=50)
raw_data.plot(start=0, duration=0.05, n_channels=1)

data = raw_data.get_data()
data.shape

import matplotlib.pyplot as plt
plt.plot(data[0][:25])

all_events[0]

np.stack(difficulties)
[x.shape[0] for x in difficulties]

raw_data.info
difficulties

events
raw_data.plot_psd(fmax=50)
raw_data.plot(duration=5, n_channels=30)

# %%
# ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
# ica.fit(raw_data)
# ica.exclude = [1, 2]  # details on how we picked these are omitted here
# ica.plot_properties(raw, picks=ica.exclude)

ds_events = dict()
for channel in raw_data.ch_names:
    # https://mne.tools/stable/generated/mne.find_events.html
    ds_events[channel] = mne.find_events(raw_data, channel)
ds_events

raw_data.info
raw_data.ch_names

data = raw_data.get_data()

raw_data.info