import numpy as np
from pydantic import BaseModel
from src.models import Model
import pandas as pd


class MetricsReport(BaseModel):
    accuracy: float


class Metrics:
    def __init__(self, model: Model, y: pd.Series) -> None:
        self.model = model
        self.y = y

    def calculate(self):
        return MetricsReport(accuracy=np.mean(self.model.predict() == self.y))


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
