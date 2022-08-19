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
from sklearn.model_selection import train_test_split
from .train_or_test_dataset import TrainOrTestDataset


class Dataset:
    """dataset has a list of Observations and methods to deal with them: loading and transforming"""

    def __init__(self, config: DatasetConfig, transformers=List[Transformer], label: str = "response_type"):
        self.config = config
        self.transformers = transformers
        self.label = label

        self.X: Union[List[Observation], np.array] = None
        self.y: pd.Series = None

    @property
    def train(self) -> TrainOrTestDataset:
        return TrainOrTestDataset(X=self.X, y=self.y, is_train=True)

    @property
    def test(self) -> TrainOrTestDataset:
        return TrainOrTestDataset(X=self.X, y=self.y, is_train=False)

    @timeit
    def load(self):
        self.X: List[Observation] = []  # makes this method idempotent

        with Pool(config.CPUs) as executor:
            products = list(self.config.combinations)
            results = executor.imap(self._load_observations, products)
            for result in results:
                self.X += result

        self._extract_labels()
        self._initialize_train_test_subsets()

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
        self.y = ys.map(getattr(MLMappings, pluralize(self.label))).to_numpy()

    @timeit
    def transform(self) -> None:
        for transformer in self.transformers:
            self.X = transformer.transform(self.X)

    def _initialize_train_test_subsets(self):
        np.random.seed(0)
        # stratification by label and person_id - we want to be able to predict for new patients
        stratify = list(zip([observation.person_id for observation in self.X], self.y))
        TrainOrTestDataset.train_samples_idxs, TrainOrTestDataset.test_sample_idxs = train_test_split(
            np.arange(len(self.y)), test_size=0.2, stratify=stratify
        )
