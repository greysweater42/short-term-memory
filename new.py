from src.dataset import Dataset


ds = Dataset()
data, dy = ds.get_data(
    exp_types=["M"],
    exp_times=[5],
    phases=["encoding"],
    electrodes=["Fz", "Fp1"],
    domain="freq",
    epoch_length=1
)
data[0]
dy[0]

import matplotlib.pyplot as plt
plt.plot(data[0][0]['freq'], data[0][0]['Fz'])
plt.plot(data[0][1]['freq'], data[0][1]['Fz'])
plt.plot(data[0][2]['freq'], data[0][2]['Fz'])
plt.ylim(top=0.01, bottom=0)
plt.show()

# n_persons = data["person"].max()
# test_persons = set(np.random.choice(range(n_persons), test_size, replace=False))
# return test_persons
