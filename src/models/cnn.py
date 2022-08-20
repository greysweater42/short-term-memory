import torch
import torch.nn as nn
from src.dataset.sub_dataset import SubDatasetLoader, SubDataset
from src.models import Model
from tqdm import tqdm
from dataclasses import dataclass


class CNN1DNet(nn.Module):
    """specification of 1-dimensional convolutional neural network: layers and classifier"""

    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 1, 10, stride=5),
            nn.Conv1d(1, 1, 10, stride=5),
        )
        self.classifier = nn.Sequential(nn.Linear(9, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.cnn(x)
        return self.classifier(x)


@dataclass
class NeuralNetwork:
    net: nn.Module
    device: str
    optimizer: torch.optim
    criterion: torch.nn.modules.loss
    batch_size: int
    epochs: int


class TorchCNNModel(Model):
    """the place where neural network (which is a bunch of parameters) meets the dataset; it can train and predict"""

    def __init__(self, neural_network: NeuralNetwork) -> None:
        self.nn = neural_network
        self.nn.net.to(self.nn.device)

    def train(self, sub_dataset: SubDataset):
        for _ in tqdm(range(self.nn.epochs), desc="epoch", position=1):
            loader = SubDatasetLoader(sub_dataset, batch_size=self.nn.batch_size, device=self.nn.device)
            for inputs, labels in tqdm(loader, desc="training", position=0):
                self._learn_one_batch(inputs, labels)

    def _learn_one_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        """neural network learns a little from given inputs and labels; by "learning" it is meant: updates parameters"""
        outputs = self.nn.net(inputs)
        loss = self.nn.criterion(outputs, labels)
        self.nn.optimizer.zero_grad()
        loss.backward()
        self.nn.optimizer.step()

    def predict(self, sub_dataset: SubDataset):
        preds = []
        loader = SubDatasetLoader(sub_dataset, batch_size=self.nn.batch_size, device=self.nn.device)
        for inputs, _ in tqdm(loader):
            preds.append(self._predict_one_batch(inputs))
        return torch.concat(preds)

    def _predict_one_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        return (self.nn.net(inputs).detach().to("cpu") > 0.5).double()
