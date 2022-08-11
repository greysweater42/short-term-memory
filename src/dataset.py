from cmath import exp
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


from dataclasses import dataclass

with open("src/params.json") as f:
    params = json.load(f)


@dataclass
class DatasetParameters:
    # TODO parametes validation
    experiment_types: List[str] = ["M", "R"]
    loads: List[int] = [5, 6, 7]
    response_types: List[str] = ["correct", "error"]
    phases: List[str] = ["encoding", "delay"]
    electrodes: List[str] = params["ELECTRODES"]


@dataclass
class ObservationConfig:
    experiment_type: str
    load: int
    response_type: str
    phase: str
    person: int = None  # for most of the time we do not need person's id; we filter by other features
    trial: int = None  # similar to person

    @property
    def path(self) -> Path:
        return (
            Path()
            / str(self.person if self.person else "*")
            / self.experiment_type
            / str(self.load)
            / self.response_type
            / str(self.trial if self.trial else "*")
            / self.phase
        ).with_suffix(".feather")

    @classmethod
    def from_path(cls, path: Path):
        person, experiment_type, load, response_type, trial, phase = path.woth_suffix("").parts
        return cls(
            experiment_type=experiment_type,
            load=int(load),
            response_type=response_type,
            trial=int(trial),
            phase=phase,
            person=int(person),
        )

    def __repr__(self):
        return f"""Observation
    person: {self.person}
    experiment type: {self.experiment_type}
    load: {self.load}
    response type: {self.response_type}
    trial: {self.trial}
    phase: {self.phase}
    electrodes: {self.electrodes}
        """


class Observation(Signal):
    def __init__(self, observation_config: ObservationConfig) -> None:
        super().__init__()
        self.observation_config: ObservationConfig = observation_config
        self.electrodes: List[str] = None
        self.data: pd.DataFrame = None

    def __repr__(self):
        return (
            self.observation
            + f"""
    example data:
    {self.data.loc[:5]}
    """
        )

    def _make_parent_dir(self):
        self.observation_data.path.parent.mkdir(parents=True, exist_ok=True)

    def write_data(self, df):
        self._make_parent_dir()
        feather.write_feather(df, self.path)

    def read_data(self, electrodes):
        self.data = feather.read_feather(self.observation_config.path)
        self.data["time"] = np.arange(0, len(self.data) / 500, 0.002)
        # self.data = self.data[electrodes]  # TODO this should not be filtered, unless for efficiency reasons
        # returns self, so it can be run asynchronously
        return self


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


class DatasetLoader:
    """loads the data for given parameters"""

    def __init__(self) -> None:
        self.dataset_parameters: DatasetParameters = None
        self.observations: List[Observation] = None

    def load(self, dataset_parameters: DatasetParameters):
        self.dataset_parameters = dataset_parameters
        self._create_observation_instances()
        self._read_data()

    def _create_observation_instances(self):
        observations = []
        combinations = product(self.experiment_types, self.loads, self.response_types, self.phases)
        for experiment_type, load, response_type, phase in combinations:
            observation_config = ObservationConfig(
                experiment_type=experiment_type, load=load, response_type=response_type, phase=phase
            )
            paths = Path(DATA_CACHE_PATH).rglob(observation_config.path)
            obcs = [ObservationConfig.from_path(path) for path in paths]
            observations += [Observation(obc) for obc in obcs]
        self.observations = observations

    def _read_data(self):
        with Pool(processes=6) as executor:
            results = executor.imap(
                partial(self._async_read_data_for_observation, electrodes=self.electrodes), self.observations
            )
            observations = []
            for result in tqdm(results, desc="loading data", total=len(self.observations)):
                # each observation runs "read_data" method, which saves data internally to its instance and returns
                # itself
                observations.append(result)
            self.observations = observations

    @staticmethod
    def _async_read_data_for_observation(observation: Observation, electrodes: List[str]):
        return observation.read_data(electrodes)


class Dataset:
    """dataset is a list of Observations, which can be concatenated, and with methods on this concatenated data"""

    def __init__(self, parameters: DatasetParameters):
        self.parameters = parameters
        self.dataset_loader: DatasetLoader = DatasetLoader()

        self.observations: List[Observation] = None
        self.is_phases_leveled = None
        self.is_concat_phases = None
        self.is_wavelet_transformed = None
        self.is_fourier_transformed = None
        self.train = None
        self.val = None
        self.labels = None
        self.label_dict = dict()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def load(self):
        self.dataset_loader.load()
        self.observations = self.dataset_loader.observations

    def create_labels(self, dimension: str):
        str_labels = [observation.__dict__[dimension] for observation in self.observations]
        self.parameters
        labels_name = dimension + "s"  # TODO smells bad

        self.labels = np.array([s == labels_name for s in str_labels]).astype(int)
        labels = zip(reversed(labels_names), range(len(labels_names)))
        self.labels_dict = {label: num for label, num in labels}

    def write_data_cache(self, cpus: int = None):
        if not cpus:
            cpus = cpu_count() // 2
        ps = [int(p.stem[4:]) for p in DATA_RAW_PATH.iterdir() if p.stem.startswith("sub")]
        persons = sorted(ps)
        if DATA_CACHE_PATH.exists():
            shutil.rmtree(DATA_CACHE_PATH)
        Path.mkdir(DATA_CACHE_PATH)

        with Pool(cpus) as executor:
            results = executor.imap(_load_clean_write_raw_data, persons)
            [_ for _ in tqdm(results, total=len(persons))]

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


loader = DatasetLoader(DatasetParameters(experiment_type=["M"], loads=[5, 6]))
loader.load()

dataset = Dataset(params=DatasetParameters())
dataset.load()
