from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
import pywt


class Transform:
    def __init__(self):
        self.data = None
        self.fouriers = dict()
        self.wavelets = dict()

    def fourier_transform(self):
        for c in self.data.columns[self.data.dtypes != "object"]:
            y = self.data[c].to_numpy()
            n = len(y)
            coefficients = 2 / n * np.abs(fft(y)[: n // 2])
            frequencies = fftfreq(n, 0.002)[: n // 2]
            self.fouriers[c] = dict()
            self.fouriers[c]["c"] = coefficients
            self.fouriers[c]["f"] = frequencies

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
        x = self.moving_average(np.abs(c), smooth)
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
    def moving_average(y, w):
        return np.array([np.convolve(x, np.ones(w), "valid") / w for x in y])
