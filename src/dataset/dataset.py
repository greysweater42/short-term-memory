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
from .sub_dataset.sub_dataset import TrainDataset, TestDataset, SubDataset


class Dataset:
    """dataset has a list of Observations and methods to deal with them: loading and transforming;
    gives acces for train and test subsets: train.X, train.y, test.X, test.y """

    def __init__(self, config: DatasetConfig, transformers=List[Transformer], label: str = "response_type") -> None:
        self.config = config
        self.transformers = transformers
        self.label = label

        self.X: Union[List[Observation], np.array] = None
        self.y: pd.Series = None

    @property
    def train(self) -> TrainDataset:
        return TrainDataset(X=self.X, y=self.y)

    @property
    def test(self) -> TestDataset:
        return TestDataset(X=self.X, y=self.y)

    @timeit
    def load(self):
        self._load_observations()
        self._extract_labels()
        SubDataset.initialize_train_test_subsets(X=self.X, y=self.y)

    def _load_observations(self) -> None:
        self.X: List[Observation] = []  # makes this method idempotent

        with Pool(config.CPUs) as executor:
            combinations = list(self.config.combinations)
            results = executor.imap(self._load_observations_for_combination, combinations)
            for result in results:
                self.X += result

    @staticmethod
    def _load_observations_for_combination(combination: Tuple) -> List[Observation]:
        observation_path = Observation.path_for_searching(
            experiment_type=combination[0],
            num_letters=combination[1],
            response_type=combination[2],
            name=combination[3],
        )
        paths = Path().rglob(str(observation_path))
        return [Observation.from_path(path) for path in paths]

    @timeit
    def _extract_labels(self) -> None:
        # TODO to transformer
        ys = pd.Series([getattr(observation, self.label) for observation in self.X])
        self.y = ys.map(getattr(MLMappings, pluralize(self.label))).to_numpy()

    @timeit
    def transform(self) -> None:
        for transformer in self.transformers:
            self.X, self.y = transformer.transform(self.X, self.y)
