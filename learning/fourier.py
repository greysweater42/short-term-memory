# %%
import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

# %% data definition
half_cycle = np.array([np.pi / 4, np.pi / 2, np.pi * 3 / 4, np.pi])
x_t = np.concatenate([half_cycle + np.pi * i for i in range(10)])
y = np.round(3 * np.sin(x_t) + np.sin(2 * x_t), 2) + 3
n = len(y)
plt.plot(x_t, y)
# %% fourier transform
F = fft(y)
F_freq = fftfreq(n, 0.125)  # half_cycle has length of 4, full cycle = 8
plt.plot(F_freq[: n // 2], 2 / n * np.abs(F[: n // 2]))
# %% inverse fourier transform
all(np.isclose(y, ifft(F).real))

# %% removing noise 2 Hz = sin(2x)
F[np.abs(F_freq) == 2] = 0
plt.plot(F_freq[: n // 2], 2 / n * np.abs(F[: n // 2]))
all(np.isclose(ifft(F).real, y - np.sin(2 * x_t)))
