from typing import List, Dict

from pydantic import BaseModel, root_validator
from src.dataset_info import DatasetInfo
from itertools import product


class MLMappings:
    """integer values are used for machine learning models, which require numeric values"""

    experiment_types: Dict[str, int] = {"M": 1, "R": 0}
    response_types: Dict[str, int] = {"correct": 1, "wrong": 0}


class DatasetConfig(BaseModel):
    """special class for parameter validation. this class could ba part of Dataset, but this solution is a little
    cleaner"""

    experiment_types: List[str] = ["M", "R"]
    num_letters: List[int] = [5, 6, 7]
    response_types: List[str] = ["correct", "wrong"]
    phases: List[str] = ["delay"]
    electrodes: List[str] = DatasetInfo.electrodes

    @root_validator
    def check_values(cls, values: Dict) -> int:
        for name, value in values.items():
            for subvalue in value:
                proper_values = DatasetInfo.get(name)
                if subvalue not in proper_values:
                    raise ValueError(f"{name} must be from the list {proper_values}; {name} is not")
        return values

    @property
    def combinations(self) -> product:
        return product(self.experiment_types, self.num_letters, self.response_types, self.phases)
