from src.dataset.dataset import Dataset
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch
from tqdm import tqdm
from torchvision import models


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, wavelets, labels):
        self.x = torch.tensor(wavelets.astype(np.float32))
        self.y = torch.tensor(labels.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].repeat(3, 1, 1), self.y[idx]


ds_t = EEGDataset(wavelets[t_wavelets], labels[t_wavelets])
dl_t = torch.utils.data.DataLoader(ds_t, batch_size=20, shuffle=True)
ds_v = EEGDataset(wavelets[v_wavelets], labels[v_wavelets])
dl_v = torch.utils.data.DataLoader(ds_v, batch_size=20, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sigmoid()

    def forward(self, x):
        return self.model(x)


class CNNModel:
    def __init__(self) -> None:
        self.device = "cuda"
        self.net = Net()
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.criterion = nn.BCELoss()

    def fit(self):
        EPOCH_NUM = 5
        for i in tqdm(range(EPOCH_NUM), desc="epoch", position=1):
            correct_t = 0
            for inputs, labels in tqdm(dl_t, desc="training", position=0):
                outputs = self.net(inputs.to(self.device))
                loss = self.criterion(outputs.view(-1), labels.float().to(self.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pred = outputs.view(-1).detach().to("cpu") > 0.5
                correct_t += sum(pred == labels.to("cpu"))
            print(f"\ntraining accuracy: {correct_t / len(ds_t) * 100}%")

            correct_v = 0
            for inputs, labels in tqdm(dl_v, desc="validation", position=0):
                outputs = self.net(inputs.to(self.device))
                pred = outputs.view(-1).detach().to("cpu") > 0.5
                correct_v += sum(pred == labels.to("cpu"))
            print(f"\nvalidation accuracy: {correct_v / len(ds_v) * 100}%")
