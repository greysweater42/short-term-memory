import numpy as np


class TrainAndTestDataset:

    train_samples_idxs: np.array = None
    test_sample_idxs: np.array = None
    is_train: bool = None

    def __init__(self, X: np.array, y: np.array) -> None:
        self._X = X
        self._y = y

    @property
    def X(self):
        assert isinstance(self._X, np.ndarray), "you cannot use train and test before converting data to np"
        return self._X[self.train_samples_idxs] if self.is_train else self._X[self.test_sample_idxs]

    @property
    def y(self):
        assert isinstance(self._X, np.ndarray), "you cannot use train and test before converting data to np"
        return self._y[self.train_samples_idxs] if self.is_train else self._y[self.test_sample_idxs]


class TrainDataset(TrainAndTestDataset):
    is_train: bool = True


class TestDataset(TrainAndTestDataset):
    is_train: bool = False
