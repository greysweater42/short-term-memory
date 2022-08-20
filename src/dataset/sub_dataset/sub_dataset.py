import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Union
import torch

from src.observation import Observation


class SubDataset:
    """sub meaning: train or test; keeps indexes of observations, which are labeled as train or test; provides acces to
    X and y for either train or test dataset"""

    train_samples_idxs: np.ndarray = None
    test_sample_idxs: np.ndarray = None
    is_train: bool = None

    def __init__(self, X: Union[np.array, torch.Tensor], y: Union[np.array, torch.Tensor]) -> None:
        assert len(X) == len(y)
        self._X = X
        self._y = y

    def __len__(self):
        return len(self.sample_idxs)

    @classmethod
    def initialize_train_test_subsets(cls, X: List[Observation], y: np.ndarray, test_size: float = 0.2) -> None:
        """decides which observations go to the train and test subsets; keeps this info in cls.train_samples_ids and
        cls.test_sample_ids, so they are available for classes which inherit from SubDataset: TrainDataset and
        TestDataset"""
        np.random.seed(0)
        # stratification by label and person_id - we want to be able to predict for new patients
        stratify = list(zip([observation.person_id for observation in X], y))
        cls.train_samples_idxs, cls.test_sample_idxs = train_test_split(
            np.arange(len(y)), test_size=test_size, stratify=stratify
        )
        # TODO dataset is inbalanced: I should apply downsampling/upsampling (probably downsampling)

    @property
    def sample_idxs(self) -> np.ndarray:
        return self.train_samples_idxs if self.is_train else self.test_sample_idxs

    @property
    def X(self) -> Union[np.ndarray, torch.Tensor]:
        msg = "you cannot use train and test before converting data to np or torch.Tensor"
        assert isinstance(self._X, np.ndarray) or isinstance(self._X, torch.Tensor), msg
        return self._X[self.sample_idxs]

    @property
    def y(self) -> Union[np.ndarray, torch.Tensor]:
        return self._y[self.sample_idxs]


class TrainDataset(SubDataset):
    is_train: bool = True


class TestDataset(SubDataset):
    is_train: bool = False
