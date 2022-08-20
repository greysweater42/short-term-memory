from ..transformer import Transformer
from src.observation import Observation
from typing import List, Tuple
import numpy as np


class ObservationsTransformer(Transformer):
    """transformer applied independently on eeg from every given observation"""

    def transform(self, observations: List[Observation], y: np.ndarray) -> Tuple[List[Observation], np.ndarray]:
        # eegs for all the observations must be of the same length, in this case: length of the shortest eeg
        common_length = min([len(observation.eeg) for observation in observations])
        for observation in observations:
            observation.eeg = self._transform_single_channel(observation.eeg.iloc[:common_length])
        return observations, y
