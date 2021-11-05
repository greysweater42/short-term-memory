import json
import shutil
from concurrent import futures
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from tqdm import tqdm

from src.eeg_raw_dataset import DATA_RAW_PATH, EEGRawDataset
from src.transform import Transform


DATA_CACHE_PATH = Path(".data_cache")

with open("src/params.json") as f:
    params = json.load(f)


def _load_clean_write_raw_data(person: int):
    raw_ds = EEGRawDataset(person=person)
    eeg = raw_ds.load_and_clean_data()
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


class Observation(Transform):
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


class Trial(Transform):
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
        fourier_transform: bool = False,
        electrodes: List[str] = params["ELECTRODES"],
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
            for r in tqdm(res, desc="transforming wavelets"):
                r.wavelet_transform()
        if fourier_transform:
            for r in tqdm(res, desc="transforming fouriers"):
                r.fourier_transform()
        return res
