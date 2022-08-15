import pywt
import numpy as np
from typing import Union
import pandas as pd


def wavelet_transform(self, x: Union[pd.Series, pd.DataFrame], freqs: np.array = None):
    if not freqs:
        freqs = np.arange(1, 40)  # from 1 to 40Hz
    scales = 400 / freqs  # 400 specific for morlet wavelet
    coef, freq = pywt.cwt(x, scales, "morl", 0.002)
