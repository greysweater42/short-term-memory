from .sub_dataset import SubDataset
import numpy as np
from typing import Union, Tuple


class SubDatasetLoader:
    """iterator, which gives batches of SubDataset (which is either train or test), the batch in form of (X, y), for a
    given batch size, already on a given device (cpu/cuda)"""
    def __init__(self, sub_dataset: SubDataset, batch_size: int, device: str = "cpu") -> None:
        self.sub_dataset = sub_dataset
        self.device = device
        self.slices = np.array([slice(i, i + batch_size) for i in range(0, len(self.sub_dataset), batch_size)])

    def __getitem__(self, idx: Union[int, slice]) -> Tuple[np.ndarray, np.ndarray]:
        # for now iterator is enough, since we keep all the data in memory anyway; but for future uses a generator would
        # be more memory efficient
        slice_ = self.slices[idx]
        return self.sub_dataset.X[slice_].to(self.device), self.sub_dataset.y[slice_].to(self.device)

    def __len__(self) -> int:
        return len(self.slices)
