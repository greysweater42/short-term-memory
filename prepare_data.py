# %%
# https://openneuro.org/datasets/ds003655/versions/1.0.0
from src.eeg_data import ShortTermMemoryEEGRawData
import numpy as np
import pandas as pd
from pathlib import Path
import os
import ray
from tqdm import tqdm


np.random.seed(42)

PATH = Path(os.environ["HOME"]) / "doktorat" / "raw_data"
test_size = 50


@ray.remote
def prepare_person(path):
    eeg_data = ShortTermMemoryEEGRawData(person=path.name, path=PATH)
    return eeg_data.prepare_dataset(["5R", "5M"], ["delay"])


def get_data_and_labels():
    ray_objs = [prepare_person.remote(person) for person in list(PATH.glob("sub*"))]
    all_data = ray.get(ray_objs)
    ray.shutdown()

    labels = pd.concat([x[0] for x in all_data])
    data = pd.concat([x[1] for x in all_data])
    return labels, data


def choose_test_persons(data, test_size):
    """divides dataset into training and testing

    Arguments:
        data {pd.DataFrame} -- EEG data in raw format
        test_size {int} -- size of test group (e.g. 50 persons)

    Returns:
        set -- set of ids of persons chosen to be in test group
    """
    n_persons = data["person"].max()
    test_persons = set(np.random.choice(range(n_persons), test_size, replace=False))
    return test_persons


def prepare_train_test_catalogues():
    """creates the following catalogue structure in data folder:
        .
        ├── test
        │   ├── correct
        │   └── error
        └── train
            ├── correct
            └── error
    """
    sets = ["train", "test"]
    values = ["correct", "error"]
    for s in sets:
        for v in values:
            path = Path("data") / s / v
            path.mkdir(parents=True, exist_ok=True)


def divide_and_save_train_and_test_data(data, labels, test_persons):
    n_persons = data["person"].max()
    for person in tqdm(range(1, n_persons + 1)):
        sub_data = data[data.person == person]
        sub_labels = labels[labels.person == person]
        s = "test" if person in test_persons else "train"
        for trial in range(sub_data["trial"].max()):
            trial_data = sub_data[sub_data.trial == trial].drop(
                ["trial", "person", "phase"], axis=1
            )
            trial_labels = sub_labels.loc[sub_labels.trial == trial, "label"]
            v = "error" if trial_labels.iloc[0] == 0 else "correct"
            trial_data.to_csv(
                Path("data") / s / v / f"{person}_{trial}.csv", index=False
            )


def main():
    ray.init()
    labels, data = get_data_and_labels()
    ray.shutdown()
    test_persons = choose_test_persons(data=data, test_size=test_size)
    prepare_train_test_catalogues()
    divide_and_save_train_and_test_data(data, labels, test_persons)


if __name__ == '__main__':
    main()
