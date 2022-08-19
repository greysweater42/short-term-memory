from .transformer import Transformer
from typing import List
from src.observation import Observation
import pandas as pd


class ObservationsToNumpyTransformer(Transformer):
    """stacks electrode measurements on top of each other"""

    @staticmethod
    def transform(observations: List[Observation]) -> List[pd.Series]:
        eegs = [observation.eeg for observation in observations]
        return pd.concat(eegs, axis=1).to_numpy().transpose()  # each eeg in each row
