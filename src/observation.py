from pathlib import Path
from pydantic import BaseModel, validator, root_validator, ValidationError
import pandas as pd
from config import DATA_CACHE_PATH
from typing import Dict, Any
from src.dataset_info import DatasetInfo
from src.utils import pluralize
import numpy as np


class Observation(BaseModel):
    name: str
    trial_id: int
    person_id: int
    num_letters: int
    experiment_type: str
    response_type: str
    electrode: str

    eeg: pd.Series

    class Config:
        # pydantic does not allow exotic types like pd.DataFrame: it needs to be allowed in this Config class
        arbitrary_types_allowed = True

    @validator("name")
    def check_name(cls, name: str):
        if name not in DatasetInfo.phases:
            raise ValidationError(f"name should be one of {list(DatasetInfo.phases)}; {name} is none of these")
        return name

    @root_validator
    def check_list_values(cls, values: Dict[str, Any]) -> int:
        list_fields_to_validate = ["num_letters", "experiment_type", "response_type", "electrode"]
        list_values = {name: values[name] for name in values if name in list_fields_to_validate}
        for name, value in list_values.items():
            proper_values = DatasetInfo.get(pluralize(name))
            if value not in proper_values:
                raise ValidationError(f"{name} must be from the list {proper_values}; {name} is not")
        return values

    @classmethod
    def from_path(cls, path: Path):
        # TODO maybe data should be stored also for separate electrodes, as a list of observations, even without index:
        # the index is np.arange(0, len(x), 0.002) for all the cases anyway
        path_tuple = path.with_suffix("").parts
        *_, person_id, num_letters, experiment_type, response_type, trial_id, name, electrode = path_tuple
        eeg: pd.DataFrame = pd.read_feather(path, columns=[electrode])
        eeg_series = eeg[electrode]
        eeg_series.index = np.arange(0, len(eeg)) * 0.002
        return cls(
            trial_id=int(trial_id),
            person_id=int(person_id),
            num_letters=int(num_letters),
            experiment_type=experiment_type,
            response_type=response_type,
            name=name,
            electrode=electrode,
            eeg=eeg_series,
        )

    @property
    def path(self) -> Path:
        return (
            DATA_CACHE_PATH
            / str(self.person_id)
            / str(self.num_letters)
            / self.experiment_type
            / self.response_type
            / str(self.trial_id)
            / self.name
            / self.electrode
        ).with_suffix(".feather")

    @staticmethod
    def path_for_searching(
        person_id: int = None,
        num_letters: int = None,
        experiment_type: str = None,
        response_type: str = None,
        name: str = None,
        electrode: str = None,
    ):
        """used for finding trials which meet specific criterium, e.g. num_letters=6 and person="*" (all persons)"""
        # can't use cls.path because of built-in pydantic validation, which would not accept "*" for int fields
        # validation is crucial for creating flawless Phases. this could be overriden by Union[int, str], but
        # path_for_searching is actually written for different purpose than creating an instance
        return (
            DATA_CACHE_PATH
            / (str(person_id) if person_id else "*")
            / (str(num_letters) if num_letters else "*")
            / (experiment_type if experiment_type else "*")
            / (response_type if response_type else "*")
            / "*"  # we always take all the trials, since their ID is artificial
            / (name if name else "*")
            / (electrode if electrode else "*")
        ).with_suffix(".feather")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # TODO I could use some better format, maybe even mongo db with ODM< where every document would be searchable
        # by all the parameters available in self.path
        eeg = self.eeg.reset_index()
        # index is setup during loading anyway. value "1" is optimal for size of the file; feather needs at least 2
        # columns
        # eeg["index"] = 1
        eeg.to_feather(self.path)
