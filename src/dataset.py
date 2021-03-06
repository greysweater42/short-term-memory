import json
import shutil
from itertools import product
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from tqdm import tqdm

from src.eeg_raw_dataset import DATA_RAW_PATH, EEGRawDataset
from src.signal import Signal


DATA_CACHE_PATH = Path(".data_cache")

with open("src/params.json") as f:
    params = json.load(f)


def _load_clean_write_raw_data(person: int):
    raw_ds = EEGRawDataset(person=person)
    eeg = raw_ds.load_and_clean_data()
    for exp_type, times in eeg.items():
        for load, response_types in times.items():
            for response_type, trials in response_types.items():
                for trial, phases in trials.items():
                    for phase, data in phases.items():
                        ob = Observation(
                            person=person,
                            experiment_type=exp_type,
                            load=load,
                            response_type=response_type,
                            trial=trial,
                            phase=phase,
                        )
                        ob.write_data(data)


class Observation(Signal):
    def __init__(
        self, person, experiment_type, load, response_type, trial, phase
    ):
        super().__init__()
        self.person = int(person)
        self.experiment_type = experiment_type
        self.load = int(load)
        self.response_type = response_type
        self.trial = int(trial)
        self.phase = phase
        self.path = (
            DATA_CACHE_PATH
            / str(self.person)
            / self.experiment_type
            / str(self.load)
            / self.response_type
            / str(self.trial)
            / self.phase
        ).with_suffix(".feather")
        self.electrodes = None

    def __repr__(self):
        return f"""Observation
    person: {self.person}
    experiment type: {self.experiment_type}
    load: {self.load}
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
            load=self.load,
            response_type=self.response_type,
            trial=self.trial,
            phase=self.phase,
        )


class Trial(Signal):
    def __init__(self, observations, phases_limits: dict = None):
        super().__init__()
        assert len(np.unique([set(o.electrodes) for o in observations])) == 1
        assert len(np.unique([o.person for o in observations])) == 1
        assert len(np.unique([o.trial for o in observations])) == 1
        assert len(np.unique([o.response_type for o in observations])) == 1
        phases = ["baseline", "presentation", "encoding", "delay", "probe"]
        self.observations = sorted(observations, key=lambda i: phases.index(i.phase))
        self.trial = self.observations[0].trial
        self.person = self.observations[0].person
        self.response_type = self.observations[0].response_type
        self.experiment_type = self.observations[0].experiment_type
        self.load = self.observations[0].load
        self.electrodes = self.observations[0].electrodes
        raw_data = []
        for o in self.observations:
            d = o.data.copy()
            d["phase"] = o.phase
            if phases_limits:
                d = d.iloc[: phases_limits[o.phase]]
            raw_data.append(d)
        self.data = pd.concat(raw_data, ignore_index=True)

    def __getitem__(self, idx):
        return self.observations[idx]

    def __len__(self):
        return len(self.observations)

    def __repr__(self):
        return f"""{type(self).__name__}
    person: {self.person}
    experiment type: {self.experiment_type}
    load: {self.load}
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
            load=self.load,
            response_type=self.response_type,
            trial=self.trial,
        )


class Dataset:
    def __init__(self):
        self.data = []
        self.phases = []
        self.experiment_types = []
        self.loads = []
        self.response_types = []
        self.electrodes = []
        self.is_phases_leveled = None
        self.is_concat_phases = None
        self.is_wavelet_transformed = None
        self.is_fourier_transformed = None
        self.is_data_loaded = None
        self.train = None
        self.val = None
        self.labels = None
        self.label_dict = dict()

    def __repr__(self):
        # TODO with labels
        pass

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def create_labels(self, dimension):
        str_labels = [getattr(d, dimension) for d in self.data]
        labels_names = getattr(self, dimension + "s")
        self.labels = np.array([s == labels_names[0] for s in str_labels]).astype(int)
        labels = zip(reversed(labels_names), range(len(labels_names)))
        self.labels_dict = {label: num for label, num in labels}

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

    def load_data(
        self,
        experiment_types: List[str] = ["M", "R"],
        loads: List[int] = [5, 6, 7],
        response_types: List[str] = ["correct", "error"],
        phases: List[str] = ["encoding", "delay"],
        electrodes: List[str] = params["ELECTRODES"],
    ):
        self.phases = phases
        self.loads = loads
        self.experiment_types = experiment_types
        self.response_types = response_types
        self.electrodes = electrodes
        all_obs = []
        combs = product(experiment_types, loads, response_types, phases)
        for exp_type, load, response_type, phase in combs:
            regex = f"*/{exp_type}/{load}/{response_type}/*/{phase}.feather"
            paths = Path(DATA_CACHE_PATH).rglob(regex)
            obs = [Observation(*p.with_suffix("").parts[-6:]) for p in paths]
            all_obs += obs
        with Pool(processes=6) as executor:
            results = executor.imap(
                partial(self._async_load_data, electrodes=electrodes), all_obs
            )
            for r in tqdm(results, desc="loading data", total=len(all_obs)):
                self.data.append(r)
            self.is_data_loaded = True

    @staticmethod
    def _async_load_data(x, electrodes):
        return x.read_data(electrodes)

    # def unload_time_domain_data(self):
    #     for d in data:
    #         d.unload_time_domain_data()
    #     self.is_data_loaded = False

    def concat_phases(self, level_phases=False):
        phase_limits = dict()
        if level_phases:
            for phase in self.phases:
                len_phase = [len(r.data) for r in self.data if r.phase == phase]
                phase_limits[phase] = min(len_phase)
        persons = np.unique([r.person for r in self.data])
        data = []
        for p in tqdm(persons, desc="concatenating phases"):
            d_p = [r for r in self.data if r.person == p]
            trials = np.unique([p.trial for p in d_p])
            for t in trials:
                ts = Trial([o for o in d_p if o.trial == t], phase_limits)
                data.append(ts)
        self.data = data
        self.is_concat_phases = True
        self.is_phases_leveled = True

    def transform_wavelet(self):
        for d in tqdm(self.data, desc="transforming wavelets"):
            d.wavelet_transform()
        self.is_wavelet_transformed = True

    def transform_fourier(self, n):
        for d in tqdm(self.data, desc="transforming fouriers"):
            d.fourier_transform(n=n)
        self.is_fourier_transformed = True

    def process_fourier(self, *args, **kwargs):
        for d in tqdm(self.data, desc="processing fouriers"):
            d.fourier_process(*args, **kwargs)
