from typing import List

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
import config
from dataclasses import dataclass
from src.filter_out_noise import filter_out_noise

from src.trial import get_trial_response, check_is_raw_trial_valid, EVENTS as TRIAL_EVENTS
from src.phase import Phase, PHASES
from pydantic import ValidationError

# mne.set_config("MNE_LOGGING_LEVEL", "ERROR")


def write_raw_persons_to_cache():
    # TODO docstring
    # TODO this should be done asynchronously
    for path in config.DATA_RAW_PATH.iterdir():
        if path.is_dir() and path.name.startswith("sub-"):
            raw_person: RawPerson = RawPerson(person_id=int(path.name[-3:]))
            raw_person.load()
            raw_person.save_preprocessed_trials_to_cache()


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
        self.person_id = person_id
        self.eeg: pd.DataFrame = None  # time (measurement every 2ms) in rows, channels in columns
        self.events: pd.DataFrame = None
        self.phases: List[Phase] = None

    def load(self):
        """loads raw data into self.data and applies basic preprocessing"""
        self._read()
        self._apply_filters()
        self._extract_phases()

    def _read(self):
        # data are stored in several files, which are interconnected
        person_str = f"sub-{self.person_id:03}"
        path = config.DATA_RAW_PATH / person_str / "eeg"

        channels = pd.read_csv(path / RawFilenames.channels.format(person_str), sep="\t")
        eog = channels.loc[channels.type == "EOG", "name"].tolist()
        data = mne.io.read_raw_eeglab(input_fname=path / RawFilenames.eeg.format(person_str), eog=eog, preload=True)
        eeg = data.get_data()

        # the easiest way to store 3 input files is in these two variables
        self.eeg = pd.DataFrame(eeg.transpose(), columns=channels["name"])
        self.eeg.index = (np.arange(len(self.eeg)) + 1) / 500  # measurements are done every 2ms
        self.events = pd.read_csv(path / RawFilenames.events.format(person_str), sep="\t")

    def _apply_filters(self):
        """apply noise filters to all the channels"""
        # TODO wouldn't logging be better to tqdm?
        desc = f"person: {self.person_id}, applying filters"
        for channel in tqdm(self.eeg.columns, desc=desc):
            self.eeg[channel] = filter_out_noise(self.eeg[channel])

    def _extract_phases(self) -> None:
        self._add_trial_id_to_events()

        for trial_id, trial in self.events.groupby("trial_id"):
            if not check_is_raw_trial_valid(trial):
                continue

            # for convenience: it will be easier to access specific rows by their "name", not .startswith
            rows = {}
            for trial_event_name, trial_event_data in TRIAL_EVENTS.items():
                rows[trial_event_name] = trial[trial["trial_type"].str.startswith(trial_event_data)]

            num_letters = rows["baseline"]["trial_type"].iloc[0][-3]
            type_ = rows["baseline"]["trial_type"].iloc[0][-2]
            response_type = get_trial_response(rows["response"]["trial_type"].iloc[0])

            self.phases: List[Phase] = []
            for phase_name, (event_start, event_end) in PHASES:
                start = rows[event_start]["onset"].iloc[0]
                end = rows[event_end]["onset"].iloc[0]
                try:
                    phase = Phase(
                        name=phase_name,
                        trial_id=trial_id,
                        person_id=self.person_id,
                        num_letters=num_letters,
                        type_=type_,
                        response_type=response_type,
                        eeg=self.eeg[self.eeg.index.between(start, end)],
                    )
                except ValidationError:
                    continue
                self.phases.append(phase)

    def _add_trial_id_to_events(self):
        """each trial gets a unique id: an integer. trial is defined as a "start" event and all the events until the
        next "start" event"""
        self.events["trial_id"] = 0
        trial_starts_cond = self.events["trial_type"].startswith(TRIAL_EVENTS["start"])
        self.events.loc[trial_starts_cond, "trial_id"] = range(1, sum(trial_starts_cond) + 1)
        self.events["trial_id"].replace(to_replace=0, method="ffill", inplace=True)

    def save_preprocessed_phases(self):
        for phase in self.phases:
            phase.save()
