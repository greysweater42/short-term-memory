from src.dataset import Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from collections import defaultdict


ds = Dataset()
# ds.write_data_cache()

data = ds.get_data(
    exp_types=["M"],
    exp_times=[7],
    phases=["delay"],
    electrodes=["Fz"],
)

df = pd.DataFrame([d.pd_repr for d in data])
df["label"] = (df["response_type"] == "correct").astype(int)
test_persons = set(np.random.choice(range(len(df.person.unique())), 50, replace=False))


for d in tqdm(data):
    d.wavelet_transform()

wavelets = defaultdict(lambda: 0)
wrongs = defaultdict(lambda: 0)
corrects = defaultdict(lambda: 0)
for d in data:
    wavelets[d.person] += d.wavelets["Fz"][:, :1485]
    if d.response_type == "correct":
        corrects[d.person] += 1
    else:
        wrongs[d.person] += 1

performance = dict()
for person in wavelets:
    performance[person] = corrects[person] / (corrects[person] + wrongs[person])


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wavelets, performance, batch_size):
        self.batch_size = batch_size
        self.x = []
        self.y = []
        for i in wavelets.keys():
            self.x.append(wavelets[i])
            self.y.append(int(performance[i] > 0.6567))
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x, batch_y


train_wavelets = {w: wavelets[w] for w in wavelets if w not in test_persons}
test_wavelets = {w: wavelets[w] for w in wavelets if w in test_persons}
train_performance = {p: performance[p] for p in performance if p not in test_persons}
test_performance = {p: performance[p] for p in performance if p in test_persons}

test = DataGenerator(test_wavelets, test_performance, batch_size=10)
train = DataGenerator(train_wavelets, train_performance, batch_size=10)


model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-3),
    metrics=["accuracy"],
)
history = model.fit(train, validation_data=test, epochs=50)
# Epoch 50/50
# 11/11 [==============================] - 1s 50ms/step - loss: 0.6735 -
# accuracy: 0.5607 - val_loss: 0.6997 - val_accuracy: 0.5918
