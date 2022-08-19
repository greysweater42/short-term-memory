import pandas as pd
from scipy.fft import fft, fftfreq
import numpy as np
from .observations_transformer import ObservationsTransformer


class FourierTransfomer(ObservationsTransformer):
    """transforms a given signal from time domain to frequency domain using Fourier transform;
    possible postprocessing with low and high bound filters and smoothing"""

    def _transform_single_channel(self, x: pd.Series) -> pd.Series:
        # more details on fourier transform in Python:
        # https://docs.scipy.org/doc/scipy/tutorial/fft.html
        n = len(x)
        coef = 2 / n * np.abs(fft(x.to_numpy(), n=n)[: n // 2])
        freq = fftfreq(n, 0.002)[: n // 2]
        return pd.Series(coef, name=x.name, index=freq)
