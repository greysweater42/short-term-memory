import shutil
from collections import defaultdict
from pathlib import Path
from typing import List
from itertools import product
from scipy.fft import fft, fftfreq, ifft

import mne
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


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


def _recursive_defaultdict():
    return defaultdict(_recursive_defaultdict)


def _load_clean_write_raw_data(person: int):
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


class Dataset:
    def __init__(self):
        pass

    def write_data_cache(self, cpus: int = None):
        if not cpus:
            cpus = cpu_count() // 2
        ps = [
            int(p.stem[4:]) for p in DATA_RAW_PATH.iterdir() if p.stem.startswith("sub")
        ]
        persons = sorted(ps)
        if DATA_CACHE_PATH.exists():
            shutil.rmtree(DATA_CACHE_PATH)
        Path.mkdir(DATA_CACHE_PATH)

        with Pool(2) as executor:
            results = executor.imap(_load_clean_write_raw_data, persons)
            [_ for _ in tqdm(results, total=len(persons))]

    def get_data(
        self,
        exp_types: List[str] = ["M", "R"],
        exp_times: List[int] = [5, 6, 7],
        response_types: List[str] = ["correct", "error"],
        phases: List[str] = ["encoding", "delay"],
        electrodes: List[str] = ELECTRODES,
    ):
        data = []
        dictionary = []
        combs = product(exp_types, exp_times, response_types, phases)
        for exp_type, exp_time, response_type, phase in combs:
            regex = f"*/{exp_type}/{exp_time}/{response_type}/*/{phase}.feather"
            paths = Path(DATA_CACHE_PATH).rglob(regex)
            for path in paths:
                df = feather.read_feather(path)[electrodes]
                df["time"] = np.arange(0, len(df) / 500, 0.002)
                data.append(df)
                dictionary.append(path.parts[-6:])
        return data, dictionary


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
        self._remove_50_Hz_from_eeg()
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

    def _remove_50_Hz_from_eeg(self):
        freq = fftfreq(len(self._eeg), 0.002)  # 500Hz -> 0.002
        for c in self._eeg.columns:
            y_fft = fft(self._eeg[c].to_numpy()).real
            y_fft[(np.abs(freq) > 49) & (np.abs(freq) < 51)] = 0
            y_fft[(np.abs(freq) > 99) & (np.abs(freq) < 101)] = 0
            y_fft[(np.abs(freq) > 149) & (np.abs(freq) < 151)] = 0
            self._eeg[c] = ifft(y_fft).real

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
