import numpy as np
from pydantic import BaseModel
from src.models import Model
from src.dataset import Dataset, SubDataset
from typing import Union
import torch


def _flatten(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(x, np.ndarray):
        return x.flatten()
    elif isinstance(x, torch.Tensor):
        return torch.flatten(x)
    else:
        raise Exception("x must be either np.ndarray or torch.Tensor")


class MetricsReport(BaseModel):
    accuracy: float
    samples_imbalance: float


class Metrics:
    def __init__(self, model: Model, dataset: Dataset) -> None:
        self.model = model
        self.dataset = dataset

    def calculate(self):
        reports = dict()
        for name, sub_dataset in {"train": self.dataset.train, "test": self.dataset.test}.items():
            reports[name] = self._generate_report(sub_dataset=sub_dataset)
        return reports

    def _generate_report(self, sub_dataset: SubDataset) -> MetricsReport:
        y_hat = np.array(_flatten(self.model.predict(sub_dataset)))
        y = np.array(_flatten(sub_dataset.y))

        return MetricsReport(
            accuracy=np.mean(y == y_hat),
            samples_imbalance=y.mean(),
        )


# from collections import namedtuple

# from sklearn.metrics import (
#     accuracy_score,
#     auc,
#     confusion_matrix,
#     precision_score,
#     recall_score,
#     roc_curve,
# )


# def calculate_metrics(model, x_t, y_t, x_v, y_v):
#     y_t_hat = model.predict(x_t)
#     y_v_hat = model.predict(x_v)
#     y_v_hat_proba = model.predict_proba(x_v)[:, 1]
#     fpr, tpr, _ = roc_curve(y_v, y_v_hat_proba)
#     cm = confusion_matrix(y_v, y_v_hat)
#     Metrics = namedtuple(
#         "Metrics", "acc_val acc_train precision recall specificity auc"
#     )
#     Parameters = namedtuple("Parameters", "model_class")
#     return (
#         Metrics(
#             acc_val=accuracy_score(y_v, y_v_hat) * 100,
#             acc_train=accuracy_score(y_t, y_t_hat) * 100,
#             precision=precision_score(y_v, y_v_hat),
#             recall=recall_score(y_v, y_v_hat),
#             specificity=cm[0][0] / sum(cm[0]),
#             auc=auc(fpr, tpr),
#         ),
#         Parameters(model_class=type(model).__name__),
#     )
