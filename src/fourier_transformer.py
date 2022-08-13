import pandas as pd
from scipy.fft import fft, fftfreq
import numpy as np
from typing import Union, Tuple


class FourierTransfomer:
    """transforms a given signal from time domain to frequency domain using Fourier transform;
    possible postprocessing with low and high bound filters and smoothing"""

    def __init__(
        self, postprocess: bool = False, bounds: Tuple[int, int] = None, smooth: int = None, stride: int = None
    ) -> None:
        self.postprocess: bool = postprocess
        self.bounds: Tuple[int, int] = bounds
        self.smooth: int = smooth
        self.stride: int = stride

    def transform(self, x: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(x, pd.Series):
            return self._transform_single_channel(x)
        elif isinstance(x, pd.DataFrame):
            transformed = []
            for channel in x.columns:
                transformed.append(self._transform_single_channel(x[channel]))
            return pd.concat(transformed)
        else:
            return Exception("signal should be of type pd.Series or pd.DataFrame")

    def _transform_single_channel(self, x: pd.Series) -> pd.Series:
        # more details on fourier transform in Python:
        # https://docs.scipy.org/doc/scipy/tutorial/fft.html
        n = len(x)
        coef = 2 / n * np.abs(fft(x.to_numpy(), n=n)[: n // 2])
        freq = fftfreq(n, 0.002)[: n // 2]
        if self.postprocess:
            coef, freq = self._postprocess_fourier(coef, freq)
        return pd.Series(coef, name=x.name, index=freq)

    def _postprocess_fourier(self, coef: np.array, freq: np.array) -> Tuple[np.array, np.array]:
        """signal in freq domain may have weird values for vary low and high frequencies and can be quite bumpy:
        this function applies low and high bound filtering, and moving average for smoothing"""
        ix_high = np.sum(freq < self.bounds[0])  # high bound - remove low freq
        ix_low = np.sum(freq < self.bounds[1])  # low bound - remove high freq
        coefs_smooth = self._moving_average_1D(coef[ix_high:ix_low], self.smooth)
        ix_high_freq = ix_high + self.smooth // 2
        ix_low_freq = ix_low - self.smooth // 2
        freqs_smooth = freq[ix_high_freq:ix_low_freq]
        return coefs_smooth[:: self.stride], freqs_smooth[:: self.stride]

    @staticmethod
    def _moving_average_1D(x, w):
        return np.convolve(x, np.ones(w), "valid") / w
