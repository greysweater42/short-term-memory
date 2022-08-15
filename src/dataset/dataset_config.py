from itertools import product
from typing import List, Any

from pydantic import BaseModel, validator
from src.phase import PHASES

from .electrodes import ELECTRODES


class DatasetConfigProperValues:
    """possible values for each of the dataset parameters"""

    experiment_types: List[int] = ["M", "R"]
    num_letters: List[int] = [5, 6, 7]
    response_types: List[str] = ["correct", "error"]
    phases: List[str] = list(PHASES)
    electrodes: List[str] = ELECTRODES

    def __getitem__(self, key: str):
        return getattr(self, key)

    @classmethod
    def validate(cls, v: Any, field_name: str) -> Any:
        proper_values = cls[field_name]
        if v not in proper_values:
            raise ValueError(f"{field_name} must be from the list {proper_values}; {v} is not")
        return v


class DatasetConfig(BaseModel):
    """special class for parameter validation. this class could ba part of Dataset, but this solution is a little
    cleaner"""

    experiment_types: List[str] = ["M", "R"]
    num_letters: List[int] = [5, 6, 7]
    response_types: List[str] = ["correct", "error"]
    phases: List[str] = ["delay"]
    electrodes: List[str] = ELECTRODES

    # TODO I wonder if there is an easier way to validate those values, maybe even without pydantic, since this
    # "validator" decorator results in a lot of duplication
    @validator("experiment_types", each_item=True)
    def check_experiment_type(cls, v: int) -> int:
        return DatasetConfigProperValues.validate(v, "experiment_types")

    @validator("num_letters", each_item=True)
    def check_num_letters(cls, v: int) -> int:
        return DatasetConfigProperValues.validate(v, "num_letters")

    @validator("response_types", each_item=True)
    def check_response_types(cls, v: str) -> str:
        return DatasetConfigProperValues.validate(v, "response_types")

    @validator("phases", each_item=True)
    def check_phases(cls, v: str) -> str:
        return DatasetConfigProperValues.validate(v, "phases")

    @validator("electrodes", each_item=True)
    def check_electrodes(cls, v: str) -> str:
        return DatasetConfigProperValues.validate(v, "electrodes")

    @property
    def combinations(self) -> product:
        return product(self.experiment_types, self.num_letters, self.response_types, self.phases)
