import pywt
import numpy as np
import pandas as pd
from .observations_transformer import ObservationsTransformer


class WaveletTransformer(ObservationsTransformer):
    """wavelet transform transforms a signal from 1-dimensional time domain to 2-dimensional domain: time x frequency;
    it shows how various frequencies were appearing/disappearing over time"""

    def _transform_single_channel(self, x: pd.Series, freqs: np.array = np.array([6, 10.5, 19, 42.5])) -> pd.DataFrame:
        scales = 400 / freqs  # 400 specific for morlet wavelet
        coef: np.ndarray
        freq: np.ndarray
        coef, freq = pywt.cwt(x, scales, "morl", 0.002)
        return pd.DataFrame(coef, index=freq.round(2), columns=x.index)
