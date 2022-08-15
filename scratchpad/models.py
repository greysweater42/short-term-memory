
# class SpectralAnalysis:
#     def __init__(self, file) -> None:
#         self.data = pd.read_csv(file)
#         self.F, self.F_freq = np.array([]), np.array([])
#         self.n = len(self.data)

#     def calculate_fourier(self):
#         # coś jest nie tak z danymi, np F3 ucina się wcześnie
#         y = self.data.loc[:, "Fz"].to_numpy()
#         n = len(y)
#         self.F = fft(y)
#         self.F_freq = fftfreq(n, 0.002)[: n // 2]  # 500Hz -> 0.002

#     def plot_spectral_eeg(self):
#         if not any(self.F):
#             raise Exception("you have to calculate_fourier before plotting")
#         plt.plot(self.F_freq[:300], 2 / self.n * np.abs(self.F[:300]))
#         plt.ylim(top=0.00002, bottom=0)
#         plt.show()

