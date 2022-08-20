from src.dataset import Dataset, DatasetConfig
import logging
from src.models.torch_cnn import TorchCNNModel, CNN1DNetSpec, TorchCNNNet
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
dataset = Dataset(config=dataset_config, transformers=transformers)
dataset.load()
dataset.transform()

net = TorchCNNNet(
    spec=CNN1DNetSpec, device="cpu", optimizer=torch.optim.Adam, criterion=nn.BCELoss, learning_rate=0.1, batch_size=10
)
model = TorchCNNModel(net=net, dataset=dataset)
model.train()

metrics = Metrics(model, dataset)
print(metrics.calculate())
