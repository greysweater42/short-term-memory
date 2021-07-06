import numpy as np
import matplotlib.pyplot as plt

x0 = np.sin(np.arange(0, 100, 0.01) * 1)
x1 = np.sin(np.arange(0, 100, 0.01) * 1/2)
x2 = np.sin(np.arange(0, 100, 0.01) * 1/4)
x3 = np.sin(np.arange(0, 100, 0.01) * 1/8)
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


plt.plot(spectrogram[:,0])
plt.plot(spectrogram[:,1])
spectrogram[0]

spectrogram[:,0]
spectrogram[:,1]


np.fft.fft(y)[:10]
np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))

sig = np.exp(2j * np.pi * np.arange(8) / 8)
np.fft.fft(sig)

import matplotlib.pyplot as plt
t = np.arange(256)
sp = np.fft.fft(np.sin(t))
freq = np.fft.fftfreq(t.shape[-1])
plt.plot(freq, sp.real)

len(freq)
sp.real
sp.imag