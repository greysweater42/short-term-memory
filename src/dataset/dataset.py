from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd

import config
from src.phase import Phase
from src.transformers import Transformer

from .dataset_config import DatasetConfig


class Dataset:
    """dataset has a list of Phases and methods to deal with them: loading and transforming"""

    def __init__(self, config: DatasetConfig, transformers=List[Transformer]):
        self.config: DatasetConfig = config
        self.transformers: List[Transformer] = transformers
        self.X: Union[List[Phase], np.array] = None
        self.y: pd.Series = None

    def load(self):
        self.X: List[Phase] = []  # makes this method idempotent

        # TODO asynchronously
        for type_, num_letters, response_type, phase in self.config.combinations:
            phase_path = Phase.path_for_searching(
                type_=type_, num_letters=num_letters, response_type=response_type, name=phase
            )
            paths = Path(config.DATA_CACHE_PATH).rglob(phase_path)
            self.X += [Phase.from_path(path) for path in paths]
            # TODO add self.y here
            self.y = "?"

    def transform(self) -> None:
        for transformer in self.transformers:
            self.X = transformer.transform(self.X)
