import logging

from src.dataset import Dataset, DatasetConfig, DatasetLoader
from src.metrics import Metrics
from src.models import XGBClassifierModel
from src.transformers import FourierTransfomer, FrequencyTransformer, LabelTransformer, EEGToNumpyTransformer



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
    LabelTransformer(),
    FourierTransfomer(),
    FrequencyTransformer(freqs_to_remove=[(0, 10), (50, 500)]),
    EEGToNumpyTransformer(),
]
dataset = Dataset(loader=DatasetLoader(config=dataset_config), transformers=transformers)
dataset.load()
dataset.transform()

model = XGBClassifierModel()
model.train(dataset.train)

metrics = Metrics(model, dataset)
metrics.calculate()
