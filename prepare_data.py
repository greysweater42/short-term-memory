# %%
# https://openneuro.org/datasets/ds003655/versions/1.0.0
from eeg_data import EEGData, PATH
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
# %%


sep_labels = []
sep_data = []

for path in tqdm(list(PATH.glob("sub*"))):
    eeg_data = EEGData(person=path.name)
    person_labels, person_data = eeg_data.prepare_dataset("5R")
    sep_labels.append(person_labels)
    sep_data.append(person_data)

data = pd.concat(sep_data)
labels = pd.concat(sep_labels)
# %%
data.to_csv("data.csv", index=False)
labels.to_csv("labels.csv", index=False)

# %%
np.random.seed(42)
test_size = 50
n_persons = data["person"].max()
test_persons = set(np.random.choice(range(n_persons), test_size, replace=False))
train_persons = set(range(n_persons)) - test_persons

# %%
sets = ["train", "test"]
values = ["correct", "error"]
for s in sets:
    for v in values:
        path = Path("data") / s / v
        path.mkdir(parents=True, exist_ok=True)

# %%
for person in tqdm(range(1, n_persons + 1)):
    sub_data = data[data.person == person]
    sub_labels = labels[labels.person == person]
    s = "test" if person in test_persons else "train"
    for trial in range(sub_data["trial"].max()):
        trial_data = sub_data[sub_data.trial == trial].drop(["trial", "person"], axis=1)
        trial_labels = sub_labels.loc[sub_labels.trial == trial, "label"]
        v = "error" if trial_labels.iloc[0] == 0 else "correct"
        trial_data.to_csv(Path("data") / s / v / f"{person}_{trial}.csv")
