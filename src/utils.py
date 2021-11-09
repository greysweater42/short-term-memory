import numpy as np
from src.dataset import Dataset


class KFolder:
    def __init__(self, ds: Dataset, k: int) -> None:
        self.ds = ds
        persons = np.arange(1, len(np.unique([d.person for d in ds])) + 1)
        np.random.shuffle(persons)
        self.folds = np.array_split(persons, k)

    def __repr__(self):
        persons = len(np.unique(np.concatenate(self.folds)))
        return f"""K-folder with {len(self.folds)} folds on {persons} persons.
        Fold sizes: {[len(f) for f in self.folds]}"""

    def __len__(self):
        return len(self.folds)

    def __iter__(self):
        for fold in self.folds:
            val_idx = set(fold)
            train_idx = set(np.concatenate(self.folds)) - val_idx
            train = [d for d in self.ds if d.person in train_idx]
            val = [d for d in self.ds if d.person in val_idx]
            yield train, val
