from typing import List, Union
import numpy as np
import pandas as pd

from src.observation import Observation
from src.transformers import Transformer

from .dataset_config import DatasetConfig, MLMappings
from .dataset_loader import DatasetLoader
from src.utils import pluralize, timeit
from .sub_dataset.sub_dataset import TrainDataset, TestDataset, SubDataset


class Dataset:
    """dataset has a list of Observations and methods to deal with them: loading and transforming;
    gives acces for train and test subsets: train.X, train.y, test.X, test.y"""

    def __init__(
        self,
        loader: DatasetLoader,
        transformers=List[Transformer],
        label: str = "response_type",
    ) -> None:
        self.loader = loader
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
        self.X = self.loader.load()
        self._extract_labels()
        SubDataset.initialize_train_test_subsets(X=self.X, y=self.y)

    @timeit
    def _extract_labels(self) -> None:
        # TODO to transformer
        ys = pd.Series([getattr(observation, self.label) for observation in self.X])
        self.y = ys.map(getattr(MLMappings, pluralize(self.label))).to_numpy()

    @timeit
    def transform(self) -> None:
        for transformer in self.transformers:
            self.X, self.y = transformer.transform(self.X, self.y)
