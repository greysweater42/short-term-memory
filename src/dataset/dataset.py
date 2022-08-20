from typing import List, Union

import numpy as np
import torch
import pandas as pd
from src.observation import Observation
from src.transformers import Transformer, LabelTransformer
from src.utils import timeit

from .dataset_loader import DatasetLoader
from .sub_dataset.sub_dataset import SubDataset, TestDataset, TrainDataset


class Dataset:
    """dataset has a list of Observations and methods to deal with them: loading and transforming;
    gives access to train and test subsets: train.X, train.y, test.X, test.y"""

    def __init__(self, loader: DatasetLoader, transformers=List[Transformer]) -> None:
        self.loader = loader
        self.transformers = transformers

        self.X: Union[List[Observation], np.ndarray, torch.Tensor] = None
        self.y: Union[pd.Series, np.ndarray, torch.Tensor] = None

    @property
    def train(self) -> TrainDataset:
        return TrainDataset(X=self.X, y=self.y)

    @property
    def test(self) -> TestDataset:
        return TestDataset(X=self.X, y=self.y)

    @timeit
    def load(self):
        self.X = self.loader.load()

    @timeit
    def transform(self) -> None:
        for transformer in self.transformers:
            self.X, self.y = transformer.transform(self.X, self.y)
            if isinstance(transformer, LabelTransformer):
                # now that we have labels ready we can decide which of the observations will belong to test set
                SubDataset.initialize_train_test_subsets(X=self.X, y=self.y)
