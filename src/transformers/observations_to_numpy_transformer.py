from .transformer import Transformer
from typing import List
from src.observation import Observation
import pandas as pd
import numpy as np


class ObservationsToNumpyTransformer(Transformer):

    @staticmethod
    def transform(observations: List[Observation], y: np.ndarray) -> List[pd.Series]:
        eegs = [observation.eeg for observation in observations]
        # each eeg in each row
        return pd.concat(eegs, axis=1).to_numpy().transpose(), y
