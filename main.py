from src.dataset import Dataset
from src.metrics import calculate_metrics
import numpy as np
import mlflow
from collections import defaultdict
from tqdm import tqdm

from xgboost import XGBClassifier


def _recursive_defaultdict():
    return defaultdict(_recursive_defaultdict)


e = "Fz"
bound_high = 1
bound_low = 40
smooth = 9
stride = 4
phases = ["encoding", "delay"]
letters = [5]

ds = Dataset()
ds.load_data(
    experiment_types=["R", "M"],
    experiment_times=letters,
    phases=phases,
    electrodes=[e],
)
# ds.concat_phases()
ds.create_labels("experiment_type")
for d in tqdm(ds, desc="fourier transform"):
    n = 1550 if d.phase == "encoding" else 3300
    d.fourier_transform(n=n)
ds.process_fourier(bounds=[bound_high, bound_low], smooth=smooth, stride=stride)
# ds.train_val_divide(val_size=50)


obs = _recursive_defaultdict()
for d in ds:
    obs[d.person][d.trial][d.phase] = d

x_t = []
y_t = []
x_v = []
y_v = []

all_obs = range(1, len(set([d.person for d in ds])) + 1)
vals = set(np.random.choice(all_obs, 50, replace=False))

for person in obs:
    for trial in obs[person]:
        encoding = obs[person][trial]["encoding"].fouriers[e].processed[0].coef_
        delay = obs[person][trial]["delay"].fouriers[e].processed[0].coef_
        x = np.concatenate((encoding, delay))
        y = int(obs[person][trial]["encoding"].experiment_type == "R")
        if person in vals:
            x_v.append(x)
            y_v.append(y)
        else:
            x_t.append(x)
            y_t.append(y)


x_t = np.array(x_t)
x_v = np.array(x_v)
y_t = np.array(y_t)
y_v = np.array(y_v)


def main(x_t, y_t, x_v, y_v):
    model = XGBClassifier(
        eval_metric="logloss",
        max_depth=2,
        subsample=0.7,
        reg_lambda=10,
        reg_alpha=10,
        colsample_bytree=0.3,
        gamma=10,
    )
    model.fit(x_t, y_t)
    return calculate_metrics(model, x_t, y_t, x_v, y_v)


with mlflow.start_run():
    metrics, parameters = main(x_t, y_t, x_v, y_v)
    metrics
    tags = {
        "class": "RM",
        "data": "fourier",
        "phases": phases,
        "letters": [5],
    }
    mlflow.set_tags(tags)
    mlflow.log_param("model class", parameters.model_class)
    mlflow.log_metric("acc_train", metrics.acc_train)
    mlflow.log_metric("acc_val", metrics.acc_val)
    mlflow.log_metric("precision", metrics.precision)
    mlflow.log_metric("recall", metrics.recall)
    mlflow.log_metric("specificity", metrics.specificity)
    mlflow.log_metric("auc", metrics.auc)
