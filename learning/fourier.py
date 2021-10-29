import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
half_cycle = np.array([np.pi / 4, np.pi / 2, np.pi * 3 / 4, np.pi])
x_t = np.concatenate([half_cycle + np.pi * i for i in range(10)])

y = np.round(3 * np.sin(x_t) + np.sin(2 * x_t), 2)
plt.plot(x_t, y)
n = len(y)
d = 10

F = fft(y)
F_freq = fftfreq(n, 0.125)[: n // 2]  # half_cycle has length of 4, full cycle = 8
F_freq
F
plt.plot(F_freq, 2 / n * np.abs(F[: n // 2]))
F_freq

# removing noise 2 Hz = sin(2x)
F = fft(y)
F
F_freq = fftfreq(n, 0.125)  # half_cycle has length of 4, full cycle = 8
F_freq
plt.plot(F_freq, np.abs(F))
F[np.abs(F_freq) == 2] = 0
plt.plot(F_freq, np.abs(F))

ifft(F).real
plt.plot(x_t, ifft(F).real)
plt.plot(x_t, y - np.sin(2 * x_t))
