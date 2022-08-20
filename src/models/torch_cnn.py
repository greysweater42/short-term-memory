import torch
import torch.nn as nn
from src.dataset import Dataset
from src.dataset.sub_dataset import SubDatasetLoader
from src.dataset.sub_dataset.sub_dataset import SubDataset
from src.models import Model
from tqdm import tqdm


class CNN1DNetSpec(nn.Module):
    """specification of 1-dimensional convolutional neural network: layers and classifier"""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 1, 10, stride=5),
            nn.Conv1d(1, 1, 10, stride=5),
        )
        self.classifier = nn.Sequential(nn.Linear(9, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.cnn(x)
        return self.classifier(x)


class TorchCNNNet:
    """neural network, bunch of parameters used for training with "learn" method"""

    def __init__(
        self,
        spec: nn.Module,
        device: str,
        optimizer: torch.optim,
        criterion: torch.nn.modules.loss,
        learning_rate: float,
        batch_size: int,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.spec = spec()
        self.spec.to(self.device)
        self.optimizer = optimizer(self.spec.parameters(), lr=learning_rate)
        self.criterion = criterion()

    def learn(self, inputs: torch.Tensor, labels: torch.Tensor):
        """neural network learns a little from given inputs and labels; by "learning" it is meant: updates parameters"""
        outputs = self.spec(inputs)
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return (self.spec(inputs).detach().to("cpu") > 0.5).double()


class TorchCNNModel(Model):
    def __init__(self, net: TorchCNNNet, dataset: Dataset) -> None:
        self.net = net
        self.dataset = dataset

    def train(self):
        for _ in tqdm(range(1), desc="epoch", position=1):
            loader = SubDatasetLoader(self.dataset.train, batch_size=self.net.batch_size, device=self.net.device)
            for inputs, labels in tqdm(loader, desc="training", position=0):
                self.net.learn(inputs, labels)

    def predict(self, sub_dataset: SubDataset):
        preds = []
        loader = SubDatasetLoader(sub_dataset, batch_size=self.net.batch_size, device=self.net.device)
        for inputs, _ in tqdm(loader):
            preds.append(self.net.predict(inputs))
        return torch.concat(preds)
