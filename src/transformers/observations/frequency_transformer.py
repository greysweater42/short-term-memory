
import pandas as pd
import numpy as np
from typing import List, Tuple
from .observations_transformer import ObservationsTransformer


class FrequencyTransformer(ObservationsTransformer):
    def __init__(self, freqs_to_remove: List[Tuple[int, int]] = None, smooth: int = 1, stride: int = 1) -> None:
        # TODO describe parameters: freqs_to_remove, smooth and stride
        self.freqs_to_remove = freqs_to_remove
        self.smooth = smooth
        self.stride = stride

    def _transform_single_channel(self, x: pd.Series) -> pd.Series:
        """signal in freq domain may have weird values for vary low and high frequencies and can be quite bumpy:
        this function applies low and high bound filtering, and moving average for smoothing"""
        self.x = x
        self.apply_freqs_to_remove()
        self.apply_smooth()
        self.apply_stride()
        return self.x

    def apply_freqs_to_remove(self):
        if self.freqs_to_remove:
            for low, high in self.freqs_to_remove:
                self.x = self.x[~self.x.index.to_series().between(low, high)]

    def apply_smooth(self):
        self.x = self._moving_average_1D(self.x, self.smooth)

    def apply_stride(self):
        self.x = self.x[:: self.stride]

    @staticmethod
    def _moving_average_1D(x: pd.Series, w: int) -> pd.Series:
        return pd.Series(np.convolve(x, np.ones(w), "valid") / w, index=x.index)