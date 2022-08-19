from pathlib import Path
from typing import List, Union, Tuple
import numpy as np
import pandas as pd

import config
from src.observation import Observation
from src.transformers import Transformer

from .dataset_config import DatasetConfig, MLMappings
from multiprocessing import Pool
from src.utils import pluralize, timeit


class Dataset:
    """dataset has a list of Observations and methods to deal with them: loading and transforming"""

    def __init__(self, config: DatasetConfig, transformers=List[Transformer], label: str = "response_type"):
        self.config = config
        self.transformers = transformers
        self.label = label

        self.X: Union[List[Observation], np.array] = None
        self.y: pd.Series = None

    @timeit
    def load(self):
        self.X: List[Observation] = []  # makes this method idempotent

        with Pool(config.CPUs) as executor:
            products = list(self.config.combinations)
            results = executor.imap(self._load_observations, products)
            for result in results:
                self.X += result

        self._extract_labels()

    @staticmethod
    def _load_observations(product: Tuple) -> List[Observation]:
        observation_path = Observation.path_for_searching(
            experiment_type=product[0],
            num_letters=product[1],
            response_type=product[2],
            name=product[3],
        )
        paths = Path().rglob(str(observation_path))
        return [Observation.from_path(path) for path in paths]

    @timeit
    def _extract_labels(self):
        ys = pd.Series([getattr(observation, self.label) for observation in self.X])
        self.y = ys.map(getattr(MLMappings, pluralize(self.label)))

    @timeit
    def transform(self) -> None:
        for transformer in self.transformers:
            self.X = transformer.transform(self.X)
