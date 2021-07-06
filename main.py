# %%
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class EEGDataset(Dataset):
    def __init__(self, path):
        self.files = list(Path(path).rglob("*.csv"))
        self.mapping = {"error": 0, "correct": 1}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if not isinstance(idx, slice):
            return self._get_one_item(self.files[idx])
        else:
            data, labels = [], []
            for file in self.files[idx]:
                one_data, label = self._get_one_item(file)
                data.append(one_data)
                labels.append(label)
            return torch.stack(data), labels

    def _get_one_item(self, file):
        label = self.mapping[file.parent.name]
        data = pd.read_csv(file).to_numpy()[:1500].transpose().astype(np.float32)
        return torch.tensor(data), label


train_dataset = EEGDataset("/home/tomek/nauka/mne/data/train")
test_dataset = EEGDataset("/home/tomek/nauka/mne/data/test")
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv1d(22, 100, 3, stride=1)
        self.classifier = nn.Sequential(
            nn.Linear(149800, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.classifier(x.view(-1, 149800))


net = Net()

# %%
device = "cuda"
net = Net()
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()


for i in tqdm(range(3), desc="epoch", position=1):
    correct = 0
    for inputs, labels in tqdm(train_loader, desc="training", position=0):
        outputs = net(inputs.to(device))
        loss = criterion(outputs.view(-1), labels.float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += sum(outputs.view(-1).to('cpu') == labels.to('cpu'))
    print(f"\ntraining accuracy: {correct / len(train_dataset) * 100}%")

    correct = 0
    for inputs, labels in tqdm(test_loader, desc="testing"):
        outputs = net(inputs.to(device))
        correct += sum(outputs.view(-1).to('cpu') == labels.to('cpu'))
    print(f"\ntesting accuracy: {correct / len(test_dataset) * 100}%")


# %%
