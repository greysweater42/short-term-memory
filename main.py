from src.dataset import Dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow

np.random.seed(42)


e = "Fz"


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


ds = Dataset()
data = ds.get_data(
    exp_types=["R", "M"],
    exp_times=[5],
    phases=["encoding"],
    concat_phases=True,
    fourier_transform=True,
    electrodes=[e],
)

ix1 = sum(data[0].fouriers["Fz"]["f"] < 1)
ix2 = sum(data[0].fouriers["Fz"]["f"] < 40)


v_persons = np.random.choice(np.arange(1, 157), 50, replace=False)
xs_t = []
ys_t = []
xs_v = []
ys_v = []
for d in data:
    if d.person in v_persons:
        xs_v.append(moving_average(d.fouriers[e]["c"][ix1:ix2], 9)[::4])
        ys_v.append(d.experiment_type)
    else:
        xs_t.append(moving_average(d.fouriers[e]["c"][ix1:ix2], 9)[::4])
        ys_t.append(d.experiment_type)

x_t = np.array(xs_t)
x_v = np.array(xs_v)
y_t = (np.array(ys_t) == "M").astype(int)
y_v = (np.array(ys_v) == "M").astype(int)

lr = LogisticRegression()
lr.fit(x_t, y_t)

y_v_hat = lr.predict(x_v)
y_t_hat = lr.predict(x_t)

acc_val = accuracy_score(y_v, y_v_hat) * 100
acc_train = accuracy_score(y_t, y_t_hat) * 100
print(acc_val)
print(acc_train)


with mlflow.start_run():
    mlflow.log_param("model spec", "RM, logistic regression on fourier")
    mlflow.log_metric("best_acc_val", acc_val)
    mlflow.log_metric("acc_train", acc_train)
