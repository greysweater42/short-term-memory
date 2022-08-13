from typing import List
from src.phase import Phase
import pandas as pd


def merge_electrodes_for_each_phase(phases: List[Phase]) -> List[pd.Series]:
    """stacks electrode measurements on top of each other"""
    return [phase.eeg.melt()["value"] for phase in phases]
