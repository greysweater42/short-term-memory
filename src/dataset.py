import shutil
from collections import defaultdict
from pathlib import Path
from typing import List
from itertools import product
from scipy.fft import fft, fftfreq, ifft
from concurrent import futures
import matplotlib.pyplot as plt
import pywt

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
WAVES = dict(
    theta=dict(min=4, mean=6, max=8),
    alpha=dict(min=8, mean=10.5, max=13),
    beta=dict(min=16, mean=19, max=22),
    gamma=dict(min=35, mean=42.5, max=50),
)


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
                        ob = Observation(
                            person=person,
                            experiment_type=exp_type,
                            experiment_time=exp_time,
                            response_type=response_type,
                            trial=trial,
                            phase=phase,
                        )
                        ob.write_data(data)


class Wavelet:
    def __init__(self):
        self.data = None
        self.wavelets = dict()

    def wavelet_transform(self, freqs=None):
        if not freqs:
            freqs = np.arange(1, 40)  # from 1 to 40Hz
        scales = 400 / freqs  # 400 specific for morlet wavelet
        for c in self.data.columns[self.data.dtypes != "object"]:
            y = self.data[c].to_numpy()
            coefficiecnts, frequencies = pywt.cwt(y, scales, "morl", 0.002)
            self.wavelets[c] = dict()
            self.wavelets[c]["c"] = coefficiecnts
            self.wavelets[c]["f"] = frequencies

    def plot_wavelets(self, e, smooth=200, waves=False):
        c, f = self.wavelets[e]["c"].copy(), self.wavelets[e]["f"].copy()
        x = self.moving_average(np.abs(c), smooth)
        smooth_shift = smooth // 2 * 0.002
        _, ax = plt.subplots(figsize=(15, 10))
        _ = ax.contourf(
            smooth_shift + np.arange(0, 0.002 * x.shape[1], 0.002),
            f,
            x * np.expand_dims(f, 1),
        )
        if waves:
            max_x = c.shape[1] * 0.002
            kwargs = dict(linestyle="--", linewidth=1, color="red")
            for name, pos in WAVES.items():
                ax.plot([0, max_x], [pos["min"], pos["min"]], **kwargs)
                ax.text(0, pos["mean"], name, color="red")
                ax.plot([0, max_x], [pos["max"], pos["max"]], **kwargs)

        ax.set_yticks(f)
        if "phase" in self.data.columns:
            d_idx = self.data.reset_index()
            locations = d_idx.groupby("phase")["index"].agg(["min", "mean", "max"])
            locations *= 0.002
            for phase, locs in locations.iterrows():
                ax.plot([locs["min"], locs["min"]], [f[0], f[-1]], color="black")
                ax.text(locs["mean"], f[-5], phase, ha="center")
                ax.plot([locs["max"], locs["max"]], [f[0], f[-1]], color="black")
        plt.show()

    @staticmethod
    def moving_average(y, w):
        return np.array([np.convolve(x, np.ones(w), "valid") / w for x in y])


class Observation(Wavelet):
    def __init__(
        self, person, experiment_type, experiment_time, response_type, trial, phase
    ):
        super().__init__()
        self.person = int(person)
        self.experiment_type = experiment_type
        self.experiment_time = int(experiment_time)
        self.response_type = response_type
        self.trial = int(trial)
        self.phase = phase
        self.path = (
            DATA_CACHE_PATH
            / str(self.person)
            / self.experiment_type
            / str(self.experiment_time)
            / self.response_type
            / str(self.trial)
            / self.phase
        ).with_suffix(".feather")
        self.electrodes = None

    def __repr__(self):
        return f"""Observation
    person: {self.person}
    experiment type: {self.experiment_type}
    experiment time: {self.experiment_time}
    response type: {self.response_type}
    trial: {self.trial}
    phase: {self.phase}
    electrodes: {self.electrodes}
    example data:
    {self.data.loc[:5]}
        """

    def _make_parent_dir(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_data(self, df):
        self._make_parent_dir()
        feather.write_feather(df, self.path)

    def read_data(self, electrodes):
        self.data = feather.read_feather(self.path)
        self.data["time"] = np.arange(0, len(self.data) / 500, 0.002)
        self.data = self.data[electrodes]
        self.electrodes = electrodes
        return self

    @property
    def pd_repr(self):
        return dict(
            person=self.person,
            experiment_type=self.experiment_type,
            experiment_time=self.experiment_time,
            response_type=self.response_type,
            trial=self.trial,
            phase=self.phase,
        )


class Trial(Wavelet):
    def __init__(self, observations, phases_limits: dict = None):
        super().__init__()
        assert len(np.unique([o.electrodes for o in observations])) == 1
        assert len(np.unique([o.person for o in observations])) == 1
        assert len(np.unique([o.trial for o in observations])) == 1
        assert len(np.unique([o.response_type for o in observations])) == 1
        phases = ["baseline", "presentation", "encoding", "delay", "probe"]
        self.observations = sorted(observations, key=lambda i: phases.index(i.phase))
        self.trial = self.observations[0].trial
        self.person = self.observations[0].person
        self.response_type = self.observations[0].response_type
        self.experiment_type = self.observations[0].experiment_type
        self.experiment_time = self.observations[0].experiment_time
        self.electrodes = self.observations[0].electrodes
        raw_data = []
        for o in self.observations:
            d = o.data.copy()
            d["phase"] = o.phase
            if phases_limits:
                d = d.iloc[: phases_limits[o.phase]]
            raw_data.append(d)
        self.data = pd.concat(raw_data, ignore_index=True)

    def __repr__(self):
        return f"""{type(self).__name__}
    person: {self.person}
    experiment type: {self.experiment_type}
    experiment time: {self.experiment_time}
    response type: {self.response_type}
    trial: {self.trial}
    phases: {[o.phase for o in self.observations]}
    electrodes: {self.electrodes}
    example data:
    {self.data.loc[:5]}
        """

    @property
    def pd_repr(self):
        return dict(
            person=self.person,
            experiment_type=self.experiment_type,
            experiment_time=self.experiment_time,
            response_type=self.response_type,
            trial=self.trial,
        )


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

        with Pool(cpus) as executor:
            results = executor.imap(_load_clean_write_raw_data, persons)
            [_ for _ in tqdm(results, total=len(persons))]

    def get_data(
        self,
        exp_types: List[str] = ["M", "R"],
        exp_times: List[int] = [5, 6, 7],
        response_types: List[str] = ["correct", "error"],
        phases: List[str] = ["encoding", "delay"],
        concat_phases: bool = False,
        level_phases: bool = False,
        wavelet_transform: bool = False,
        electrodes: List[str] = ELECTRODES,
    ):
        all_obs = []
        combs = product(exp_types, exp_times, response_types, phases)
        for exp_type, exp_time, response_type, phase in combs:
            regex = f"*/{exp_type}/{exp_time}/{response_type}/*/{phase}.feather"
            paths = Path(DATA_CACHE_PATH).rglob(regex)
            obs = [Observation(*p.with_suffix("").parts[-6:]) for p in paths]
            all_obs += obs
        with futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(lambda ob: ob.read_data(electrodes), all_obs)
            res = [r for r in results]
        if concat_phases:
            phase_limits = dict()
            if level_phases:
                for phase in phases:
                    len_phase = [len(r.data) for r in res if r.phase == phase]
                    phase_limits[phase] = min(len_phase)
            persons = np.unique([r.person for r in res])
            data = []
            for p in persons:
                d_p = [r for r in res if r.person == p]
                trials = np.unique([p.trial for p in d_p])
                for t in trials:
                    ts = Trial([o for o in d_p if o.trial == t], phase_limits)
                    data.append(ts)
            res = data
        if wavelet_transform:
            for r in tqdm(res):
                r.wavelet_transform()
        return res


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
        for c in self._eeg.columns:
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
