from .transformer import Transformer
from typing import List
from src.observation import Observation
import pandas as pd
import numpy as np


class EEGToNumpyTransformer(Transformer):

    @staticmethod
    def transform(observations: List[Observation], y: np.ndarray) -> List[pd.Series]:
        eegs = [observation.eeg.to_numpy() for observation in observations]
        # observations in first dimension (or "rows" in 2 dimensions)
        return np.stack(eegs), y
