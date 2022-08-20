
import pandas as pd
import numpy as np
from typing import List, Tuple
from .observations_transformer import ObservationsTransformer


class FrequencyTransformer(ObservationsTransformer):
    """applies several cleaning techniques on data in frequency domain, i.e. already transformed by Fourier transform:
    removes specific frequencies (e.g. high and low), smoothenes the series with a moving average and applies stride: 
    selects every k-th element of the series, which lowers its resolution """
    def __init__(self, freqs_to_remove: List[Tuple[int, int]] = None, smooth: int = 1, stride: int = 1) -> None:
        self.freqs_to_remove = freqs_to_remove
        self.smooth = smooth
        self.stride = stride

    def _transform_single_channel(self, x: pd.Series) -> pd.Series:
        """signal in freq domain may have weird values for vary low and high frequencies and can be quite bumpy:
        this function applies low and high bound filtering, and moving average for smoothing"""
        self.x = x
        self._apply_freqs_to_remove()
        self._apply_smooth()
        self._apply_stride()
        return self.x

    def _apply_freqs_to_remove(self):
        if self.freqs_to_remove:
            for low, high in self.freqs_to_remove:
                self.x = self.x[~self.x.index.to_series().between(low, high)]

    def _apply_smooth(self):
        self.x = self._moving_average_1D(self.x, self.smooth)

    def _apply_stride(self):
        self.x = self.x[:: self.stride]

    @staticmethod
    def _moving_average_1D(x: pd.Series, w: int) -> pd.Series:
        return pd.Series(np.convolve(x, np.ones(w), "valid") / w, index=x.index)