from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, ifft
from tqdm import tqdm


mne.set_config("MNE_LOGGING_LEVEL", "ERROR")

DATA_RAW_PATH = Path("raw_data")


def _recursive_defaultdict():
    return defaultdict(_recursive_defaultdict)


class EEGRawDataset:

    files = dict(
        channels="{}_task-VerbalWorkingMemory_channels.tsv",
        eeg="{}_task-VerbalWorkingMemory_eeg.set",
        events="{}_task-VerbalWorkingMemory_events.tsv",
    )

    def __init__(self, person: int):
        self.person = person
        self.person_str = "sub-" + (3 - len(str(person))) * "0" + str(person)
        self._eeg = pd.DataFrame()
        self._events = pd.DataFrame()
        self._trials = []

    def load_and_clean_data(self):
        self._load_data()
        self._apply_filters()
        self._extract_trials()
        eeg = self._extract_eeg()
        return eeg

    def _load_data(self):
        path = DATA_RAW_PATH / self.person_str / "eeg"

        channels_path = path / self.files["channels"].format(self.person_str)
        channels = pd.read_csv(channels_path, sep="\t")
        eeg_path = path / self.files["eeg"].format(self.person_str)
        eog = channels.loc[channels.type == "EOG", "name"].tolist()
        data = mne.io.read_raw_eeglab(input_fname=eeg_path, eog=eog, preload=True)
        eeg = data.get_data()
        self._eeg = pd.DataFrame(eeg.transpose(), columns=channels["name"])

        events_path = path / self.files["events"].format(self.person_str)
        self._events = pd.read_csv(events_path, sep="\t")

    def _apply_filters(self):
        freq = fftfreq(len(self._eeg), 0.002)  # 500Hz -> 0.002
        desc = f"person: {self.person}, applying filters"
        for c in tqdm(self._eeg.columns, desc=desc):
            y_fft = fft(self._eeg[c].to_numpy()).real
            # 50Hz line filter
            y_fft[(np.abs(freq) > 45) & (np.abs(freq) < 51)] = 0
            y_fft[(np.abs(freq) > 99) & (np.abs(freq) < 101)] = 0
            y_fft[(np.abs(freq) > 149) & (np.abs(freq) < 151)] = 0
            # low-pass filter
            y_fft[(np.abs(freq) > 45)] = 0
            # high-pass filter
            y_fft[(np.abs(freq) < 2)] = 0
            self._eeg[c] = ifft(y_fft).real

    @staticmethod
    def _filter_func(start, end):
        return lambda x: x.startswith(start) and x.endswith(end)

    def _extract_trials(self):
        ps = [
            self._events["trial_type"].str.startswith("start of the baseline period"),
            self._events["trial_type"].str.startswith("presentation"),
            self._events["trial_type"].str.startswith("encoding"),
            self._events["trial_type"].str.startswith("delay"),
            self._events["trial_type"].str.startswith("probe"),
            self._events["trial_type"].str.startswith("Response"),
        ]
        ps_idx = [np.array(p[p].index) for p in ps]
        orders = np.array(
            [
                np.isin((ps_idx[0] + 1), ps_idx[1]),
                np.isin((ps_idx[0] + 2), ps_idx[2]),
                np.isin((ps_idx[0] + 3), ps_idx[3]),
                np.isin((ps_idx[0] + 4), ps_idx[4]),
                np.isin((ps_idx[0] + 5), ps_idx[5]),
            ]
        )
        props = ps_idx[0][orders.sum(0) == len(orders)]
        self._trials = pd.DataFrame(
            dict(
                time=self._events.loc[props + 2, "trial_type"].str[-2].to_numpy(),
                type=self._events.loc[props + 2, "trial_type"].str[-1].to_numpy(),
                baseline=self._events.loc[props, "onset"].to_numpy(),
                presentation=self._events.loc[props + 1, "onset"].to_numpy(),
                encoding=self._events.loc[props + 2, "onset"].to_numpy(),
                delay=self._events.loc[props + 3, "onset"].to_numpy(),
                probe=self._events.loc[props + 4, "onset"].to_numpy(),
                response=self._events.loc[props + 5, "onset"].to_numpy(),
                response_type=self._events.loc[props + 5, "trial_type"].to_numpy(),
            )
        )
        self._trials["response_type"] = self._trials["response_type"].map(
            lambda x: "correct" if "correct" in x else "error"
        )
        self._trials["time"] = self._trials["time"].astype(int)

    def _extract_eeg(self):
        continuous_time = (np.arange(len(self._eeg)) + 1) / 500
        phases = {
            "baseline": ["baseline", "presentation"],
            "presentation": ["presentation", "encoding"],
            "encoding": ["encoding", "delay"],
            "delay": ["delay", "probe"],
            "probe": ["probe", "response"],
        }
        eegs = _recursive_defaultdict()
        for phase, (start, end) in phases.items():
            from_ = self._trials[start]
            to = self._trials[end]
            intervals = pd.IntervalIndex.from_arrays(from_, to)
            cuts = pd.cut(continuous_time, intervals)
            cuts.categories = [str(i) for i in range(len(self._trials))]
            self._eeg["trial"] = cuts
            eeg = self._eeg[cuts.notna()]
            for trial, data in eeg.groupby("trial"):
                ti = int(trial)
                e = self._trials.loc[ti]
                d_eeg = data.drop("trial", axis=1)
                eegs[e["type"]][e["time"]][e["response_type"]][ti][phase] = d_eeg

        return eegs
