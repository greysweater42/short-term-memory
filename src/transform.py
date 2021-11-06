from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
import pywt
import json
from collections import defaultdict
from reprlib import Repr


# TODO move to a .py file
with open("src/params.json") as f:
    params = json.load(f)


def _recursive_defaultdict():
    return defaultdict(_recursive_defaultdict)


class Transform:
    def __init__(self):
        self.data = None
        self.fouriers = dict()
        self.wavelets = dict()

    def fourier_transform(self, n):
        for c in self.data.columns[self.data.dtypes != "object"]:
            x = self.data[c].to_numpy()
            f = Fourier(n=n)
            f.transform(x)
            self.fouriers[c] = f

    def fourier_process(self, *args, **kwargs):
        for fourier in self.fouriers.values():
            fourier.process(*args, **kwargs)

    def plot_fourier(self, e):
        ix = sum(self.fouriers[e]["f"] < 50)
        plt.plot(self.fouriers[e]["f"][ix], self.fouriers[e]["c"][ix])

    def wavelet_transform(self, freqs=None):
        if not freqs:
            freqs = np.arange(1, 40)  # from 1 to 40Hz
        scales = 400 / freqs  # 400 specific for morlet wavelet
        for c in self.data.columns[self.data.dtypes != "object"]:
            y = self.data[c].to_numpy()
            coefficiecnts, frequencies = pywt.cwt(y, scales, "morl", 0.002)
            self.wavelets[c] = dict()
            self.wavelets[c]["c"] = coefficiecnts
            self.wavelets[c]["f"] = frequencies

    def plot_wavelet(self, e, smooth=200, waves=False):
        c, f = self.wavelets[e]["c"].copy(), self.wavelets[e]["f"].copy()
        x = self.moving_average_2D(np.abs(c), smooth)
        smooth_shift = smooth // 2 * 0.002
        _, ax = plt.subplots(figsize=(15, 10))
        _ = ax.contourf(
            smooth_shift + np.arange(0, 0.002 * x.shape[1], 0.002),
            f,
            x * np.expand_dims(f, 1),
        )
        if waves:
            max_x = c.shape[1] * 0.002
            kwargs = dict(linestyle="--", linewidth=1, color="red")
            for name, pos in params["WAVES"].items():
                ax.plot([0, max_x], [pos["min"], pos["min"]], **kwargs)
                ax.text(0, pos["mean"], name, color="red")
                ax.plot([0, max_x], [pos["max"], pos["max"]], **kwargs)

        ax.set_yticks(f)
        if "phase" in self.data.columns:
            d_idx = self.data.reset_index()
            locations = d_idx.groupby("phase")["index"].agg(["min", "mean", "max"])
            locations *= 0.002
            for phase, locs in locations.iterrows():
                ax.plot([locs["min"], locs["min"]], [f[0], f[-1]], color="black")
                ax.text(locs["mean"], f[-5], phase, ha="center")
                ax.plot([locs["max"], locs["max"]], [f[0], f[-1]], color="black")
        plt.show()

    @staticmethod
    def moving_average_2D(y, w):
        return np.array([np.convolve(x, np.ones(w), "valid") / w for x in y])


class Fourier:
    def __init__(self, n: int) -> None:
        self.freq_ = np.array([])
        self.coef_ = np.array([])
        self.processed = []
        self.n = n

    def __repr__(self):
        r = Repr().repr(list(self.x))
        return f"""Fourier transform for
        {r}
        transformed: {True if any(self.freq_) else False}
        num processed: {len(self.processed)}
        """

    def transform(self, x):
        self.x = x
        self.coef_ = 2 / self.n * np.abs(fft(x, n=self.n)[: self.n // 2])
        self.freq_ = fftfreq(self.n, 0.002)[: self.n // 2]

    def process(self, *args, **kwargs):
        fp = FourierProcessed(*args, **kwargs)
        fp.transform(coef=self.coef_, freq=self.freq_)
        self.processed.append(fp)

    def plot(self):
        pass


class FourierProcessed:
    def __init__(self, bounds, smooth, stride) -> None:
        self.bounds = bounds
        self.smooth = smooth
        self.stride = stride
        self.coef_ = np.array([])
        self.freq_ = np.array([])

    def __repr__(self):
        return f"""Processed Fourier with parameters:
        bounds: {self.bounds}
        smooth: {self.smooth}
        stride: {self.stride}

        processed: {True if any(self.coef_) else False}
        """

    def transform(self, coef, freq):
        ix_high = sum(freq < self.bounds[0])  # high bound - remove low freq
        ix_low = sum(freq < self.bounds[1])  # low bound - remove high freq
        coefs_smooth = self.moving_average_1D(coef[ix_high:ix_low], self.smooth)
        ix_high_freq = ix_high + self.smooth // 2
        ix_low_freq = ix_low - self.smooth // 2
        freqs_smooth = freq[ix_high_freq:ix_low_freq]
        self.coef_ = coefs_smooth[:: self.stride]
        self.freq_ = freqs_smooth[:: self.stride]

    @staticmethod
    def moving_average_1D(x, w):
        return np.convolve(x, np.ones(w), "valid") / w
