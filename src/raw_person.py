from typing import List, Dict, Union

import mne
import numpy as np
import pandas as pd
import config
from dataclasses import dataclass
from src.filter_out_noise import filter_out_noise

from src.trial import Trial, EVENTS
from src.phase import Phase

import warnings
from .utils import timeit
from multiprocessing import Pool
from functools import partial

warnings.filterwarnings("ignore", category=DeprecationWarning)


def write_raw_persons_to_cache():
    # TODO docstring
    # TODO there should be a console API (in click) to run this function
    # TODO this api could also have a function for downloading data
    for path in config.DATA_RAW_PATH.iterdir():
        if path.is_dir() and path.name.startswith("sub-"):
            raw_person: RawPerson = RawPerson(person_id=int(path.name[-3:]))
            raw_person.load()
            raw_person.save_preprocessed_phases()


@dataclass
class RawFilenames:
    """data is stored in a few files:
    - channels describe the electrodes (there are dozens of them during the analysis)
    - eeg are the signals (brainwaves)
    - events are the labels, i.e. information about whether the was any stimulus applied to the patient and when
    """

    channels: str = "{}_task-VerbalWorkingMemory_channels.tsv"
    eeg: str = "{}_task-VerbalWorkingMemory_eeg.set"
    events: str = "{}_task-VerbalWorkingMemory_events.tsv"


class RawPerson:
    """load, preprocess and save in friendlier format raw data of a single person who took part in study.
    some vocabulary may be useful:
    - channel - an electrode placed on skull. There are typically 21 electrodes placed like in this picture
    https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29#/media/File:21_electrodes_of_International_10-20_system_for_EEG.svg
    - trial - part of the survey when a person is e.g. presented with letters and tries to memorize it and answer; there
    are many trials for a person
    - phase - part of a trial when a person either is preparing, reads the letters, tries to memorize them etc.
    """

    def __init__(self, person_id: int):
        self.person_id: int = person_id
        self.eeg: pd.DataFrame = None  # time (measurement every 2ms) in rows, channels in columns
        self.events: pd.DataFrame = None
        self.phases: List[Phase] = None

    @timeit
    def load(self):
        """loads raw data into self.data and applies basic preprocessing"""
        self._read()
        self._apply_filters()
        self._extract_phases_from_trials()

    @timeit
    def _read(self):
        # data are stored in several files, which are interconnected
        person_str = f"sub-{self.person_id:03}"
        path = config.DATA_RAW_PATH / person_str / "eeg"

        channels = pd.read_csv(path / RawFilenames.channels.format(person_str), sep="\t")
        eog = channels.loc[channels.type == "EOG", "name"].tolist()
        data = mne.io.read_raw_eeglab(input_fname=path / RawFilenames.eeg.format(person_str), eog=eog, preload=True)
        eeg = data.get_data()

        # the easiest way to store 3 input files is in these two variables
        self.eeg = pd.DataFrame(eeg.transpose(), columns=channels["name"].to_list())
        self.eeg.index = (np.arange(len(self.eeg)) + 1) / 500  # measurements are done every 2ms
        events_path = path / RawFilenames.events.format(person_str)
        self.events = pd.read_csv(events_path, sep="\t", index_col="onset", usecols=["onset", "trial_type"])

    @timeit
    def _apply_filters(self):
        filtered_eeg: List[pd.Series] = []
        with Pool(config.CPUs) as executor:
            results = executor.map(filter_out_noise, [self.eeg[channel] for channel in self.eeg.columns])
            for result in results:
                filtered_eeg.append(result)
        self.eeg = pd.concat(filtered_eeg, axis=1)

    @timeit
    def _extract_phases_from_trials(self) -> None:
        self._add_trial_id_to_events()
        self.phases: List[Phase] = []

        all_kwargs: List[Dict[str, Union[str, int, pd.DataFrame]]] = []
        for trial_id, trial_df in self.events.groupby("trial_id"):
            start = trial_df.index.min()
            end = trial_df.index.max()
            eeg = self.eeg[self.eeg.index.to_series().between(start, end, inclusive="left")]
            all_kwargs.append(dict(person_id=self.person_id, eeg=eeg, trial_df=trial_df, trial_id=trial_id))
        with Pool(config.CPUs) as executor:
            trials_tasks = [
                executor.apply_async(partial(Trial.extract_phases_from_args, **kwargs)) for kwargs in all_kwargs
            ]
            for trial_task in trials_tasks:
                self.phases += trial_task.get()

    def _add_trial_id_to_events(self):
        """each trial gets a unique id: an integer. trial is defined as a "start" event and all the events until the
        next "start" event"""
        self.events["trial_id"] = 0
        trial_starts_cond = self.events["trial_type"].str.startswith(EVENTS["start"])
        self.events.loc[trial_starts_cond, "trial_id"] = range(1, sum(trial_starts_cond) + 1)
        self.events["trial_id"].replace(to_replace=0, method="ffill", inplace=True)

    def save_preprocessed_phases(self):
        for phase in self.phases:
            phase.save()
