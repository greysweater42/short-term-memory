import matplotlib.pyplot as plt
import pywt
import numpy as np
import pandas as pd


dataset = "http://paos.colorado.edu/research/wavelets/wave_idl/sst_nino3.dat"
df_nino = pd.read_table(dataset)
N = df_nino.shape[0]
t0 = 1871
dt = 0.25
time = np.arange(0, N) * dt + t0
signal = df_nino.values.squeeze()
scales = np.arange(1, 128)

[coefficients, frequencies] = pywt.cwt(signal, scales, "cmor", dt)
power = (abs(coefficients)) ** 2
period = 1.0 / frequencies
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
contourlevels = np.log2(levels)

fig, ax = plt.subplots(figsize=(15, 10))
im = ax.contourf(
    time,
    np.log2(period),
    np.log2(power),
    contourlevels,
    extend="both",
    cmap=plt.cm.seismic,
)

ax.set_title("Wavelet Transform (Power Spectrum) of signal", fontsize=20)
ax.set_ylabel("Period (years)", fontsize=18)
ax.set_xlabel("Time", fontsize=18)

yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
ax.set_yticks(np.log2(yticks))
ax.set_yticklabels(yticks)
ax.invert_yaxis()
ylim = ax.get_ylim()
ax.set_ylim(ylim[0], -1)

cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
fig.colorbar(im, cax=cbar_ax, orientation="vertical")
plt.show()
