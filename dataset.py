import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from collections import namedtuple


class Dataset:

    def __init__(self, path, ds_types):
        self.path = path
        self.ds_types = ds_types

    @staticmethod
    def calculate_fourier_for_path(path):
        sa = SpectralAnalysis(path)
        sa.calculate_fourier()
        values = sa.F.real[2:10]
        label = path.parent.stem
        return label, values

    def prepare_datasets(self):
        for ds_type in self.ds_types:
            ds = self._prepare_dataset(ds_type)
            setattr(self, ds_type, ds)

    def _prepare_dataset(self, ds_type):
        print(f"Preparing {ds_type} dataset")
        ys = []
        labels = []
        paths = list((self.path / ds_type).rglob("*.csv"))
        with Pool(6) as executor:
            results = executor.imap(self.calculate_fourier_for_path, paths)
            for result in tqdm(results, total=len(paths)):
                label, y = result
                labels.append(label)
                ys.append(y)

        X = pd.DataFrame(ys)
        y = np.array([1 if lb == "correct" else 0 for lb in labels])
        ds = namedtuple(ds_type, 'X y')
        return ds(X=X, y=y)


class SpectralAnalysis:
    def __init__(self, file) -> None:
        self.data = pd.read_csv(file)
        self.F, self.F_freq = np.array([]), np.array([])
        self.n = len(self.data)

    def __repr__(self):
        return f"""Spectral Analysis for data:
        {self.data.iloc[:5,:5]}
        length: {self.n}
        fourier representation: {"calculated" if any(self.F) else "not calculated"}"""

    def calculate_fourier(self):
        # coś jest nie tak z danymi, np F3 ucina się wcześnie
        y = self.data.loc[:, "Fz"].to_numpy()
        n = len(y)
        self.F = fft(y)
        self.F_freq = fftfreq(n, 0.002)[: n // 2]  # 500Hz -> 0.002

    def plot_spectral_eeg(self):
        if not any(self.F):
            raise Exception("you have to calculate_fourier before plotting")
        plt.plot(self.F_freq[:300], 2 / self.n * np.abs(self.F[:300]))
        plt.ylim(top=0.00002, bottom=0)
        plt.show()


