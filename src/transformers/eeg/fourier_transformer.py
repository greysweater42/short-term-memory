import pandas as pd
from scipy.fft import fft, fftfreq
import numpy as np
from .eeg_transformer import EEGTransformer


class FourierTransfomer(EEGTransformer):
    """transforms a given eeg signal from time domain to frequency domain using Fourier transform"""

    def _transform_single_channel(self, x: pd.Series) -> pd.Series:
        # more details on fourier transform in Python:
        # https://docs.scipy.org/doc/scipy/tutorial/fft.html
        n = len(x)
        coef = 2 / n * np.abs(fft(x.to_numpy(), n=n)[: n // 2])
        freq = fftfreq(n, 0.002)[: n // 2]
        return pd.Series(coef, name=x.name, index=freq)
