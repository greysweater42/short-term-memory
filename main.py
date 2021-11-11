from src.dataset import Dataset
from src.metrics import calculate_metrics
import numpy as np
import mlflow
from collections import defaultdict


def _recursive_defaultdict():
    return defaultdict(_recursive_defaultdict)


e = ["Fz", "P3", "P4"]
bound_high = 1
bound_low = 40
smooth = 9
stride = 4
phases = ["delay"]
letters = [5]
label = "response_type"

ds = Dataset()
ds.load_data(
    experiment_types=["R", "M"],
    loads=letters,
    phases=phases,
    electrodes=e,
)
# ds.concat_phases()
ds.transform_fourier(n=3300)
ds.process_fourier(bounds=[bound_high, bound_low], smooth=smooth, stride=stride)

new_ds = []

for person in range(1, 157):
    dp = [d for d in ds if d.person == person]
    errors = [d for d in dp if d.response_type == "error"]
    corrects = np.random.choice([d for d in dp if d.response_type == "correct"], len(errors), replace=False)
    new_ds += errors
    new_ds += corrects.tolist()


ds.data = new_ds

ds.create_labels(label)
ds.train_val_divide(val_size=50)

x_t = []
x_v = []
y_t = []
y_v = []
for ll, d in ds.train:
    fs = []
    for el in e:
        fs.append(d.fouriers[el].processed[0].coef_)
    x_t.append(np.concatenate(fs))
    y_t.append(ll)

for ll, d in ds.val:
    fs = []
    for el in e:
        fs.append(d.fouriers[el].processed[0].coef_)
    x_v.append(np.concatenate(fs))
    y_v.append(ll)

x_t = np.array(x_t)
x_v = np.array(x_v)
y_t = np.array(y_t)
y_v = np.array(y_v)


def main(x_t, y_t, x_v, y_v):
    from xgboost import XGBClassifier

    model = XGBClassifier(max_depth=1)
    model.fit(x_t, y_t)
    return calculate_metrics(model, x_t, y_t, x_v, y_v)


with mlflow.start_run():
    metrics, parameters = main(x_t, y_t, x_v, y_v)
    metrics
    tags = {
        "class": "error" if label == "response_type" else "RM",
        "data": "fourier",
        "phases": phases,
        "letters": [5],
        "electrodes": e,
        "additional": "poor tuning",
    }
    mlflow.set_tags(tags)
    mlflow.log_param("model class", parameters.model_class)
    mlflow.log_metric("acc_train", metrics.acc_train)
    mlflow.log_metric("acc_val", metrics.acc_val)
    mlflow.log_metric("precision", metrics.precision)
    mlflow.log_metric("recall", metrics.recall)
    mlflow.log_metric("specificity", metrics.specificity)
    mlflow.log_metric("auc", metrics.auc)
