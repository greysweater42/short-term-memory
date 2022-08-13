import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Tuple

import config
from tqdm import tqdm

from src.fourier_transformer import FourierTransfomer
from src.phase import Phase


# TODO params to python
with open("src/params.json") as f:
    params = json.load(f)


@dataclass
class DatasetParameters:
    # TODO parametes validation
    experiment_type: List[str] = ["M", "R"]
    num_letters: List[int] = [5, 6, 7]
    response_types: List[str] = ["correct", "error"]
    phases: List[str] = ["encoding", "delay"]
    electrodes: List[str] = params["ELECTRODES"]


class Dataset:
    """dataset is a list of Observations, which can be concatenated, and with methods on this concatenated data"""

    domain: str = None

    def __init__(self, parameters: DatasetParameters):
        self.parameters = parameters
        self.dataset_parameters: DatasetParameters = None
        self.phases: List[Phase] = None

    def load(self, dataset_parameters: DatasetParameters):
        self.dataset_parameters = dataset_parameters

        combinations = product(
            self.dataset_parameters.experiment_types,
            self.dataset_parameters.num_letters,
            self.dataset_parameters.response_types,
            self.dataset_parameters.phases,
        )
        self.phases: List[Phase] = []

        # TODO asynchronously
        for type_, num_letters, response_type, phase in combinations:
            phase_path = Phase.path_for_searching(
                type_=type_, num_letters=num_letters, response_type=response_type, name=phase
            )
            paths = Path(config.DATA_CACHE_PATH).rglob(phase_path)
            self.phases += [Phase.from_path(path) for path in paths]
        self.domain = "ms"

    def transform_fourier(
        self, postprocess: bool = False, bounds: Tuple[int, int] = None, smooth: int = None, stride: int = None
    ) -> None:
        if self.state["domain"] == "ms":
            fourier_transformer = FourierTransfomer(
                postprocess=postprocess, bounds=bounds, smooth=smooth, stride=stride
            )
            for phase in tqdm(self.phases, "transforming phases to freq domain"):
                phase.eeg = fourier_transformer.transform(phase.eeg)
            self.domain = "freq"
        else:
            # TODO more informative message
            raise WrongDomainException

    # def transform_wavelet(self):
    #     for d in tqdm(self.data, desc="transforming wavelets"):
    #         d.wavelet_transform()

    def plot(self, e):
        # TODO different plots depending on state; thic can actually be a factory
        ix = sum(self.fouriers[e]["f"] < 50)
        plt.plot(self.fouriers[e]["f"][ix], self.fouriers[e]["c"][ix])


class WrongDomainException(Exception):
    pass
