import numpy as np
import matplotlib.pyplot as plt

x0 = np.sin(np.arange(0, 100, 0.01) * 1)
x1 = np.sin(np.arange(0, 100, 0.01) * 1 / 2)
x2 = np.sin(np.arange(0, 100, 0.01) * 1 / 4)
x3 = np.sin(np.arange(0, 100, 0.01) * 1 / 8)
y = x0 + x1 + x2 + x3

plt.plot(y)

from scipy import signal

freqs, times, spectrogram = signal.spectrogram(y)

len(freqs)
times
spectrogram
plt.plot(np.round(freqs, 2))
plt.plot(np.round(times, 2))
np.round(times, 2)


plt.plot(spectrogram[:, 0])
plt.plot(spectrogram[:, 1])
spectrogram[0]

spectrogram[:, 0]
spectrogram[:, 1]


np.fft.fft(y)[:10]
np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))

sig = np.exp(2j * np.pi * np.arange(8) / 8)
np.fft.fft(sig)

import matplotlib.pyplot as plt

t = np.arange(256) / np.pi
sp = np.fft.fft(np.sin(t))
np.round(sp.real, 3)
len(sp.real)
np.round(np.sin(t), 2)
plt.plot(sp.real)

freq = np.fft.fftfreq(t.shape[-1])
plt.plot(freq, sp.real)

len(freq)
sp.real
sp.imag

# %%
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

N = 600
T = 1.0 / 800.0
x = np.linspace(0.0, N * T, N, endpoint=False)
y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)

y1 = 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
y2 = np.sin(50.0 * 2.0 * np.pi * x)
plt.plot(x[:50], y1[:50] + y2[:50])

plt.plot(x[:200], y[:200])
yf = fft(y)
xf = fftfreq(N, T)[: N // 2]
N
T
a = fftfreq(N, T)
plt.plot(yf[:5000])


plt.plot(xf, 2.0 / N * np.abs(yf[0 : N // 2]))
plt.grid()
plt.show()
yf
2.0 / N * np.abs(yf[0 : N // 2])


# %%
signal_span = np.r_[0:2*np.pi : np.pi / 4 ]
signal_in = np.sin(signal_span)

signal_in
plt.plot(signal_span, signal_in)
signal_fft = np.fft.fft(signal_in)
signal_fft_power = np.abs(signal_fft)
plt.plot(signal_fft_power, 'o')

half_cycle = np.array([np.pi/4, np.pi/2, np.pi * 3/4, np.pi])
x_t = np.concatenate([half_cycle + np.pi * i for i in range(10)])

y = np.round(3 * np.sin(x_t) + np.sin(2 * x_t), 2)
plt.plot(x_t, y)
y
n = len(y)
d = 10

F = fft(y)
F_freq = fftfreq(n, 0.125)[:n // 2]  # half_cycle has length of 4, full cycle = 8
F_freq
plt.plot(F_freq, 2 / n * np.abs(F[:n // 2]))
