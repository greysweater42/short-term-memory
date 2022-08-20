import logging

import numpy as np
import torch
import torch.nn as nn

from src.dataset import Dataset, DatasetConfig, DatasetLoader
from src.metrics import Metrics
from src.models.cnn import CNN1DNet, NeuralNetwork, TorchCNNModel
from src.transformers import (
    LabelTransformer,
    NumpyToTorchTransformer,
    EEGToNumpyTransformer,
    WaveletTransformer,
)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

dataset_config = DatasetConfig(
    experiment_types=["R", "M"],
    num_letters=[5],
    response_types=["correct"],
    phases=["delay"],
    electrodes=["T4"],
)

freqs = np.array([6, 10.5, 19, 42.5])
transformers = [
    LabelTransformer(label="experiment_type"),
    WaveletTransformer(freqs=freqs),
    EEGToNumpyTransformer(),
    NumpyToTorchTransformer(),
]
dataset = Dataset(loader=DatasetLoader(config=dataset_config), transformers=transformers)
dataset.load()
dataset.transform()

net = CNN1DNet(linear_input_channels=dataset.X.shape[-1], input_channels=len(freqs))
neural_network = NeuralNetwork(
    net=net,
    device="cpu",
    optimizer=torch.optim.Adam(net.parameters(), lr=0.1),
    criterion=nn.BCELoss(),
    batch_size=10,
    epochs=1,
)

model = TorchCNNModel(neural_network=neural_network)
model.train(dataset.train)

metrics = Metrics(model, dataset)
print(metrics.calculate())
