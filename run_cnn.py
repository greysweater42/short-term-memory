from src.dataset import Dataset, DatasetConfig, DatasetLoader
import logging
from src.models.cnn import TorchCNNModel, CNN1DNet, NeuralNetwork
from src.metrics import Metrics
import torch
import torch.nn as nn

from src.transformers import (
    FourierTransfomer,
    FrequencyTransformer,
    ObservationsToNumpyTransformer,
    NumpyToTorchTransformer,
)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

dataset_config = DatasetConfig(
    experiment_types=["R"],
    num_letters=[5],
    response_types=["wrong", "correct"],
    phases=["delay"],
    electrodes=["T4"],
)

transformers = [
    FourierTransfomer(),
    FrequencyTransformer(freqs_to_remove=[(0, 10), (50, 500)]),
    ObservationsToNumpyTransformer(),
    NumpyToTorchTransformer(),
]
dataset = Dataset(loader=DatasetLoader(config=dataset_config), transformers=transformers)
dataset.load()
dataset.transform()


net = CNN1DNet(input_channels=1)
neural_network = NeuralNetwork(
    net=net,
    device="cpu",
    optimizer=torch.optim.Adam(net.parameters(), lr=0.1),
    criterion=nn.BCELoss(),
    batch_size=10,
    epochs=1
)

model = TorchCNNModel(neural_network=neural_network)
model.train(dataset.train)

metrics = Metrics(model, dataset)
print(metrics.calculate())
