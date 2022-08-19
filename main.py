# import mlflow
from src.models import Model
from src.models.xgb import XGB
from src.dataset import DatasetConfig, Dataset
from src.transformers import FourierTransfomer, PhasesToNumpyTransformer
from src.metrics import Metrics
from dataclasses import dataclass
from typing import Optional


@dataclass
class Analysis:
    dataset: Dataset
    model: Model
    metrics: Optional[Metrics] = None

    def run(self):
        self.dataset.load()
        self.dataset.transform()
        self.model.train()
        self.metrics = Metrics(self.model, self.dataset.y)
        self.metrics.calculate()


def get_analysis() -> Analysis:
    dataset_parameters = DatasetConfig(
        experiment_types=["M", "R"],
        num_letters=[5],
        response_types=["correct", "error"],
        # TODO what happens when we apply two phases?
        phases=["delay"],
        electrodes=["Fz"],
    )
    transformers = [
        FourierTransfomer(postprocess=True, bounds=(1, 40), smooth=9, stride=4),
        PhasesToNumpyTransformer(),
        # TODO: PCATransformer
    ]

    dataset = Dataset(parameters=dataset_parameters, transformers=transformers)
    model = XGB(dataset=dataset)
    return Analysis(dataset=dataset, model=model)


if __name__ == "__main__":
    analysis = get_analysis()
    analysis.run()

# with mlflow.start_run():
# TODO check if mlflow can measure times of steps: dataset load, transform, model train and metrics
# tags = {
#     # "class": "error" if label == "response_type" else "RM",
#     "data": [tr.__class__ for tr in analysis.dataset.transformers],
#     "phases": analysis.dataset.parameters.phases,
#     "letters": analysis.dataset.parameters.num_letters,
#     "electrodes": analysis.dataset.parameters.electrodes,
#     "additional": "pca 20",
# }
# mlflow.set_tags(tags)
# mlflow.log_param("model class", analysis.parameters.model.__class__)
# mlflow.log_metric("acc_train", analysis.metrics.acc_train)
# mlflow.log_metric("acc_val", analysis.metrics.acc_val)
# mlflow.log_metric("precision", analysis.metrics.precision)
# mlflow.log_metric("recall", analysis.metrics.recall)
# mlflow.log_metric("specificity", analysis.metrics.specificity)
# mlflow.log_metric("auc", analysis.metrics.auc)
