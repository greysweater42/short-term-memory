from src.dataset import Dataset
from src.metrics import calculate_metrics
import mlflow


# %%
from src.dataset import DatasetParameters, Dataset
dataset_parameters = DatasetParameters(
    experiment_types=["M", "R"],
    num_letters=[5],
    response_types=["correct", "error"],
    phases=["delay"],
    electrodes=["Fz", "P3", "P4"]
)
ds = Dataset(parameters=dataset_parameters)
ds.load()
ds.transform_fourier(postprocess=True, bounds=(1, 40, ), smooth=9, stride=4)

from src.model_preprocess_utils import merge_electrodes_for_each_phase
import pandas as pd

eegs_flat = merge_electrodes_for_each_phase(ds.phases)
X = pd.concat(eegs_flat).transpose()  # each phase in each row
y = [phase.response_type for phase in ds.phases]


def main(X, y):
    from sklearn import svm
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = svm.SVC(probability=True)
    from sklearn.decomposition import PCA

    pca = PCA(20)
    pca.fit(X_train)
    model.fit(pca.transform(X_train), y_train)
    return calculate_metrics(model, pca.transform(X_train), y_train, pca.transform(X_test), y_test)


# with mlflow.start_run():
#     metrics, parameters = main(x_t, y_t, x_v, y_v)
#     metrics
#     tags = {
#         "class": "error" if label == "response_type" else "RM",
#         "data": "fourier",
#         "phases": phases,
#         "letters": [5],
#         "electrodes": e,
#         "additional": "pca 20",
#     }
#     mlflow.set_tags(tags)
#     mlflow.log_param("model class", parameters.model_class)
#     mlflow.log_metric("acc_train", metrics.acc_train)
#     mlflow.log_metric("acc_val", metrics.acc_val)
#     mlflow.log_metric("precision", metrics.precision)
#     mlflow.log_metric("recall", metrics.recall)
#     mlflow.log_metric("specificity", metrics.specificity)
#     mlflow.log_metric("auc", metrics.auc)
