from src.dataset import Dataset
import numpy as np
import tensorflow as tf


ds = Dataset()
# ds.write_data_cache(cpus=5)

data = ds.get_data(
    exp_types=["R"],
    exp_times=[7],
    phases=["encoding", "delay", "probe"],
    concat_phases=True,
    level_phases=True,
    wavelet_transform=True,
    electrodes=["Fz"],
)

wavelets = [np.expand_dims(d.wavelets["Fz"]["c"], 2) for d in data]
labels = [int(d.response_type == "correct") for d in data]


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wavelets, labels, batch_size=10):
        self.batch_size = batch_size
        self.x = np.array(wavelets)
        self.y = np.array(labels)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x, batch_y


te = set(np.random.choice(range(1, 157), 50, replace=False))
tr = set(range(1, 157)) - te
train = DataGenerator(np.array(wavelets)[list(tr)], np.array(labels)[list(tr)])
test = DataGenerator(np.array(wavelets)[list(te)], np.array(labels)[list(te)])

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(39, 4929, 1)
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1),
    ]
)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-3),
    metrics=["accuracy"],
)
history = model.fit(train, validation_data=test, epochs=10)
# Epoch 10/10
# 11/11 [==============================] - 3s 304ms/step - loss: 0.6144 -
# accuracy: 0.1887 - val_loss: 0.5875 - val_accuracy: 0.1000
