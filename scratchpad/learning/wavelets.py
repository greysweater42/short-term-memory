import pywt
import matplotlib.pyplot as plt
import numpy as np


# %%  the simplest case
y = np.concatenate(
    (
        np.sin(np.arange(500) / 100 * np.pi),
        np.sin(np.arange(500) / 30 * np.pi),
        np.sin(np.arange(500) / 10 * np.pi),
    )
)

dt = 0.002  # 500Hz
s = np.arange(1, 121)
coefficients, frequencies = pywt.cwt(y, s, "gaus1", dt)
plt.matshow(coefficients)

# %% simplest case - gaus1
y = [
    np.sin(np.linspace(0, 6 * 2 * np.pi, 500)),  # 6Hz  theta
    np.sin(np.linspace(0, 11 * 2 * np.pi, 500)),  # 11Hz  alpha
    np.sin(np.linspace(0, 19 * 2 * np.pi, 500)),  # 19Hz  beta
    np.sin(np.linspace(0, 40 * 2 * np.pi, 500)),  # 40Hz  gamma
]
y = np.concatenate(y)

sampling_period = 0.002
freqs = np.arange(1, 50)  # freqs we will search for
scales = 100 / freqs  # 100 / freqs are scales specifically for gaus1
coefficients, frequencies = pywt.cwt(y, scales, "gaus1", sampling_period)
assert all(freqs == np.round(frequencies).astype(int))  # for gaus1
power = coefficients ** 2
# plt.set_cmap("Reds")
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.contourf(
    np.arange(0, dt * len(y), dt),
    frequencies,
    power
    # power * np.expand_dims(frequencies, 1)
)
ax.set_yticks(frequencies)

# %% simplest case - morlet
y_raw = [
    np.sin(np.linspace(0, 6 * 2 * np.pi, 500)),  # 6Hz  theta
    np.sin(np.linspace(0, 11 * 2 * np.pi, 500)),  # 11Hz  alpha
    np.sin(np.linspace(0, 19 * 2 * np.pi, 500)),  # 19Hz  beta
    np.sin(np.linspace(0, 40 * 2 * np.pi, 500)),  # 40Hz  gamma
]
y_raw.append(y_raw[0] + y_raw[3])  # theta + gamma
y = np.concatenate(y_raw)
plt.plot(y)

sampling_period = 0.002
freqs = np.arange(1, 50)  # freqs we will search for
scales = 400 / freqs  # 400 / freqs are scales specifically for morlet
coefficients, frequencies = pywt.cwt(y, scales, "morl", sampling_period)
power = coefficients ** 2
# plt.set_cmap("Reds")
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.contourf(
    np.arange(0, dt * len(y), dt),
    frequencies,
    power * np.expand_dims(frequencies, 1)
    # power
)
ax.set_yticks(frequencies)


def moving_average(y, w):
    return np.array([np.convolve(x, np.ones(w), "valid") / w for x in y])


w = 200
power_ma = moving_average(power, 200)
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.contourf(
    w // 2 * dt + np.arange(0, dt * power_ma.shape[1], dt),
    frequencies,
    power_ma * np.expand_dims(frequencies, 1)
    # power
)
ax.set_yticks(frequencies)

# %%
