from .transformer import Transformer
from typing import List
from src.phase import Phase
import pandas as pd


class PhasesToNumpyTransformer(Transformer):
    """stacks electrode measurements on top of each other"""

    def transform(phases: List[Phase]) -> List[pd.Series]:
        stacked_phases = [phase.eeg.melt()["value"] for phase in phases]
        return pd.concat(stacked_phases).to_numpy().transpose()  # each phase in each row
