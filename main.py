from src.dataset import Dataset
from src.metrics import calculate_metrics
import numpy as np
from xgboost import XGBClassifier
import mlflow


e = "Fz"
bound_high = 1
bound_low = 40
smooth = 9
stride = 4

ds = Dataset()
ds.load_data(
    experiment_types=["R", "M"],
    experiment_times=[5],
    phases=["encoding"],
    electrodes=[e],
)
ds.concat_phases()
ds.create_labels("experiment_type")
ds.transform_fourier(n=1550)
ds.process_fourier(bounds=[bound_high, bound_low], smooth=smooth, stride=stride)
ds.train_val_divide(val_size=50)

x_t = np.array([d[1].fouriers[e].processed[0].coef_ for d in ds.train])
x_v = np.array([d[1].fouriers[e].processed[0].coef_ for d in ds.val])

y_t = np.array([d[0] for d in ds.train])
y_v = np.array([d[0] for d in ds.val])

model = XGBClassifier(max_depth=2)
model.fit(x_t, y_t)

y_t_hat = model.predict(x_t)
y_v_hat = model.predict(x_v)
y_v_hat_proba = model.predict_proba(x_v)[:, 1]

metrics = calculate_metrics(
    y_t=y_t, y_t_hat=y_t_hat, y_v=y_v, y_v_hat=y_v_hat, y_v_hat_proba=y_v_hat_proba
)

with mlflow.start_run():
    mlflow.log_param("model spec", "RM, xgb on fourier, max_depth=2")
    mlflow.log_metric("acc_train", metrics.acc_train)
    mlflow.log_metric("acc_val", metrics.acc_val)
    mlflow.log_metric("precision", metrics.precision)
    mlflow.log_metric("recall", metrics.recall)
    mlflow.log_metric("specificity", metrics.specificity)
    mlflow.log_metric("auc", metrics.auc)
