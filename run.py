from src.dataset import Dataset, DatasetConfig
import logging
from src.models.xgb import XGBClassifier
from src.metrics import Metrics

from src.transformers import FourierTransfomer, FrequencyTransformer, ObservationsToNumpyTransformer


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

dataset_config = DatasetConfig(
    experiment_types=["R"],
    num_letters=[5],
    response_types=["wrong", "correct"],
    phases=["delay"],
    electrodes=["T4"],
)

transformers = [
    FourierTransfomer(),
    FrequencyTransformer(freqs_to_remove=[(0, 10), (50, 500)]),
    ObservationsToNumpyTransformer(),
]
dataset = Dataset(config=dataset_config, transformers=transformers)
dataset.load()
dataset.transform()

model = XGBClassifier(dataset=dataset)
model.train()

metrics = Metrics(model, dataset)
metrics.calculate()
