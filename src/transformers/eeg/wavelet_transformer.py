import pywt
import numpy as np
import pandas as pd
from .eeg_transformer import EEGTransformer


class WaveletTransformer(EEGTransformer):
    """wavelet transform transforms a signal from 1-dimensional time domain to 2-dimensional domain: time x frequency;
    it shows how various frequencies were appearing/disappearing over time"""
    def __init__(self, freqs: np.array = np.array([6, 10.5, 19, 42.5])) -> None:
        super().__init__()
        self.freqs = freqs

    def _transform_single_channel(self, x: pd.Series) -> pd.DataFrame:
        scales = 400 / self.freqs  # 400 specific for morlet wavelet
        coef: np.ndarray
        freq: np.ndarray
        coef, freq = pywt.cwt(x, scales, "morl", 0.002)
        return pd.DataFrame(coef, index=freq.round(2), columns=x.index)
