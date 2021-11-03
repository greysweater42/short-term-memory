from src.dataset import Dataset
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch
from tqdm import tqdm
from torchvision import models
import logging
import sys
import mlflow
from datetime import datetime


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
hdl_file = logging.FileHandler("./logs.log")
hdl_file.setFormatter(logging.Formatter(format))
hdl_stdout = logging.StreamHandler(sys.stdout)
hdl_stdout.setFormatter(logging.Formatter(format))

logger.addHandler(hdl_file)
logger.addHandler(hdl_stdout)

torch.manual_seed(0)


def get_data(force_reload=False):
    if (
        Path("labels.npy").exists()
        and Path("wavelets.npy").exists()
        and Path("persons.npy").exists()
        and not force_reload
    ):
        with open("wavelets.npy", "rb") as f:
            wavelets = np.load(f)
        with open("labels.npy", "rb") as f:
            labels = np.load(f)
        with open("persons.npy", "rb") as f:
            persons = np.load(f)
        return wavelets, labels, persons

    ds = Dataset()
    data = ds.get_data(
        exp_types=["R", "M"],
        exp_times=[5],
        phases=["delay"],
        concat_phases=True,
        level_phases=True,
        wavelet_transform=True,
        electrodes=["T5"],
    )

    wavelets = np.array([np.expand_dims(d.wavelets["T5"]["c"], 0) for d in data])
    labels = np.array([int(d.experiment_type == "M") for d in data])
    persons = np.array([d.person for d in data])

    with open("wavelets.npy", "wb") as f:
        np.save(f, wavelets)
    with open("labels.npy", "wb") as f:
        np.save(f, labels)
    with open("persons.npy", "wb") as f:
        np.save(f, persons)

    return wavelets, labels, persons


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, wavelets, labels):
        self.x = torch.tensor(wavelets.astype(np.float32))
        self.y = torch.tensor(labels.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


wavelets, labels, persons = get_data(force_reload=True)
wavelets = wavelets[:, :, :, ::2]


np.random.seed(42)
v_persons = np.random.choice(np.arange(1, 157), 50, replace=False)
t_persons = np.array(list(set(np.arange(1, 157)) - set(v_persons)))

v_wavelets = np.isin(persons, v_persons)
t_wavelets = np.isin(persons, t_persons)

BATCH_SIZE = 10
ds_t = EEGDataset(wavelets[t_wavelets], labels[t_wavelets])
dl_t = torch.utils.data.DataLoader(ds_t, batch_size=BATCH_SIZE, shuffle=True)
ds_v = EEGDataset(wavelets[v_wavelets], labels[v_wavelets])
dl_v = torch.utils.data.DataLoader(ds_v, batch_size=BATCH_SIZE, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)

        for param in self.model.features[:-2].parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[1].in_features, 500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x.repeat(1, 3, 1, 1))


device = "cuda"
net = Net()
net.to(device)
LEARNING_RATE = 0.0001
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()


accuracy = dict(train=[], val=[])
EPOCH_NUM = 5
t0 = datetime.now()
for i in tqdm(range(EPOCH_NUM), desc="epoch", position=1):
    correct_t = 0
    for inputs, labels in tqdm(dl_t, desc="training", position=0):
        outputs = net(inputs.to(device))
        loss = criterion(outputs.view(-1), labels.float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = outputs.view(-1).detach().to("cpu") > 0.5
        correct_t += sum(pred == labels.to("cpu"))
    acc = np.round(correct_t / len(ds_t) * 100, 2)
    logger.info(f"training accuracy: {acc}%")
    accuracy["train"].append(acc)

    correct_v = 0
    for inputs, labels in tqdm(dl_v, desc="validation", position=0):
        outputs = net(inputs.to(device))
        pred = outputs.view(-1).detach().to("cpu") > 0.5
        correct_v += sum(pred == labels.to("cpu"))
    acc = np.round(correct_v / len(ds_v) * 100, 2)
    logger.info(f"validation accuracy: {acc}%")
    accuracy["val"].append(acc)
    import time
    time.sleep(20)

train_time = np.round((datetime.now() - t0).seconds / 60, 2)

ix = accuracy["val"].index(max(accuracy["val"]))
with mlflow.start_run():
    mlflow.log_param("epochs", EPOCH_NUM)
    mlflow.log_param("learning rate", LEARNING_RATE)
    mlflow.log_param("batch size", BATCH_SIZE)
    mlflow.log_param("best epoch", ix)
    mlflow.log_param("train time", train_time)
    mlflow.log_param("model spec", "RM, T5, mobilenet freeze [:-2], delay")
    mlflow.log_metric("best_acc_val", float(accuracy["val"][ix]))
    mlflow.log_metric("acc_train", float(accuracy["train"][ix]))
