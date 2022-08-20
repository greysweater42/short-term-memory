import pandas as pd
import numpy as np
from typing import List, Tuple

from src.observation import Observation
from .transformer import Transformer
from src.dataset.dataset_config import MLMappings
from src.utils import pluralize


class LabelTransformer(Transformer):
    def __init__(self, label: str = "response_type") -> None:
        self.label = label

    def transform(self, observations: List[Observation], y: np.ndarray) -> Tuple[List[Observation], np.ndarray]:
        ys = pd.Series([getattr(observation, self.label) for observation in observations])
        y_ml = ys.map(getattr(MLMappings, pluralize(self.label))).to_numpy()
        return observations, y_ml
