import shutil
from collections import defaultdict
from pathlib import Path
from typing import List
from itertools import product
from scipy.fft import fft, fftfreq

import mne
import numpy as np
import pandas as pd
import pyarrow.feather as feather


def _recursive_defaultdict():
    return defaultdict(_recursive_defaultdict)


mne.set_config("MNE_LOGGING_LEVEL", "ERROR")

DATA_CACHE_PATH = Path(".data_cache")
DATA_RAW_PATH = Path("raw_data")
ELECTRODES = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
    "EOGv",
    "EOGh",
]


class Dataset:
    def __init__(self):
        ps = [
            int(p.stem[4:]) for p in DATA_RAW_PATH.iterdir() if p.stem.startswith("sub")
        ]
        self.persons = sorted(ps)

    def write_data_cache(self):
        if DATA_CACHE_PATH.exists():
            shutil.rmtree(DATA_CACHE_PATH)
        Path.mkdir(DATA_CACHE_PATH)
        # TODO multiprocessing
        for person in self.persons:
            print(person)
            self._load_clean_write_raw_data(person=person)

    def _load_clean_write_raw_data(self, person: int):
        raw_ds = EEGRawDataset()
        eeg = raw_ds.load_and_clean_data(person=person)
        for exp_type, times in eeg.items():
            for exp_time, response_types in times.items():
                for response_type, trials in response_types.items():
                    for trial, phases in trials.items():
                        for phase, data in phases.items():
                            filename = phase + ".feather"
                            path = (
                                DATA_CACHE_PATH
                                / str(person)
                                / exp_type
                                / str(exp_time)
                                / response_type
                                / str(trial)
                            )
                            path.mkdir(parents=True, exist_ok=True)
                            feather.write_feather(data, path / filename)

    def get_data(
        self,
        exp_types: List[str] = ["M", "R"],
        exp_times: List[int] = [5, 6, 7],
        response_types: List[str] = ["correct", "error"],
        phases: List[str] = ["encoding", "delay"],
        electrodes: List[str] = ELECTRODES,
        domain: str = "time",
        epoch_length: float = 0
    ):
        assert domain in ["time", "freq"]
        data = []
        dictionary = []
        combs = product(exp_types, exp_times, response_types, phases)
        for exp_type, exp_time, response_type, phase in combs:
            regex = f"*/{exp_type}/{exp_time}/{response_type}/*/{phase}.feather"
            paths = Path(DATA_CACHE_PATH).rglob(regex)
            for path in paths:
                d = feather.read_feather(path)[electrodes]
                if epoch_length:
                    time = np.arange(0, len(d) / 500, 0.002)
                    d['epoch'] = np.floor(time / epoch_length).astype(int)
                else:
                    d["epoch"] = 0
                ds = []
                # TODO multiprocessing
                for _, dee in d.groupby('epoch'):
                    if len(d[d['epoch'] == 0]) < len(dee):  # remove last short epoch
                        continue
                    de = dee.drop("epoch", axis=1)
                    if domain == "freq":
                        de = self._transform_fourier_columnwise(de)
                    else:
                        de['time'] = np.arange(0, len(de) / 500, 0.002)
                    ds.append(de)
                data.append(ds)
                dictionary.append(path.parts[-6:])
        return data, dictionary

    @staticmethod
    def _transform_fourier_columnwise(df):
        for c in df:
            df[c] = np.abs(fft(df[c].to_numpy()).real)
        freq = fftfreq(len(df), 0.002)[: len(df) // 2]  # 500Hz -> 0.002
        df = df[: (len(df) // 2)].copy()
        df["freq"] = freq
        return df


class EEGRawDataset:

    files = dict(
        channels="{}_task-VerbalWorkingMemory_channels.tsv",
        eeg="{}_task-VerbalWorkingMemory_eeg.set",
        events="{}_task-VerbalWorkingMemory_events.tsv",
    )

    def __init__(self):
        self._eeg = pd.DataFrame()
        self._events = pd.DataFrame()
        self._trials = []

    @staticmethod
    def _make_person_str(i: int):
        if i < 10:
            return f"sub-00{i}"
        elif i < 100:
            return f"sub-0{i}"
        else:
            return f"sub-{i}"

    def load_and_clean_data(self, person: int):
        self.person = person
        self.person_str = self._make_person_str(person)
        self._load_data()
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

    @staticmethod
    def _filter_func(start, end):
        return lambda x: x.startswith(start) and x.endswith(end)

    def _extract_trials(self):
        ps = [
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
            ]
        )
        props = ps_idx[0][orders.sum(0) == 3]
        self._trials = pd.DataFrame(
            dict(
                time=self._events.loc[props, "trial_type"].str[-2].to_numpy(),
                type=self._events.loc[props, "trial_type"].str[-1].to_numpy(),
                encoding=self._events.loc[props, "onset"].to_numpy(),
                delay=self._events.loc[props + 1, "onset"].to_numpy(),
                probe=self._events.loc[props + 2, "onset"].to_numpy(),
                response=self._events.loc[props + 3, "onset"].to_numpy(),
                response_type=self._events.loc[props + 3, "trial_type"].to_numpy(),
            )
        )
        self._trials["response_type"] = self._trials["response_type"].map(
            lambda x: "correct" if "correct" in x else "error"
        )
        self._trials["time"] = self._trials["time"].astype(int)

    def _extract_eeg(self):
        continuous_time = (np.arange(len(self._eeg)) + 1) / 500
        phases = {"encoding": ["encoding", "delay"], "delay": ["delay", "probe"]}
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
