from scipy.fft import fft, fftfreq, ifft
import numpy as np


# TODO move to FourierTransformer
def filter_out_noise(eeg):
    """filters out unwanted frequencies;
    some current frequencies are typical for electricity providers and they add noise to the data;
    this function filters them out. Also high-pass and low-pass filters are applied on frequencies
    for which brainwaves do not occur"""
    freq = fftfreq(len(eeg), 0.002)  # 500Hz -> 0.002s (2ms)
    y_fft = fft(eeg).real
    # 50Hz line filter
    y_fft[(np.abs(freq) > 45) & (np.abs(freq) < 51)] = 0
    y_fft[(np.abs(freq) > 99) & (np.abs(freq) < 101)] = 0
    y_fft[(np.abs(freq) > 149) & (np.abs(freq) < 151)] = 0
    # low-pass filter
    y_fft[(np.abs(freq) > 45)] = 0
    # high-pass filter
    y_fft[(np.abs(freq) < 2)] = 0
    return ifft(y_fft).real
