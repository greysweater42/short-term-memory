from typing import List, Dict

from pydantic import BaseModel, root_validator
from src.survey_info import SurveyInfo
from itertools import product
import hashlib


class MLMappings:
    """machine learning models require numeric values, not strings; this class maps string values to numbers"""

    experiment_types: Dict[str, int] = {"M": 1, "R": 0}
    response_types: Dict[str, int] = {"correct": 1, "wrong": 0}


class DatasetConfig(BaseModel):
    """special class for parameter validation. this class could ba part of Dataset, but this solution is a little
    cleaner"""

    experiment_types: List[str] = ["M", "R"]
    num_letters: List[int] = [5, 6, 7]
    response_types: List[str] = ["correct", "wrong"]
    phases: List[str] = ["delay"]
    electrodes: List[str] = SurveyInfo.electrodes

    @root_validator
    def check_values(cls, values: Dict) -> int:
        """check if the user gave proper values of parameters and sorts them"""
        for name, value in values.items():
            for subvalue in value:
                proper_values = SurveyInfo.get(name)
                if subvalue not in proper_values:
                    raise ValueError(f"{name} must be from the list {proper_values}; {name} is not")
        # sorting is useful for consistent md5 sum
        return {k: sorted(v) for k, v in values.items()}

    @property
    def combinations(self) -> product:
        """all possible combinations of parameters of DatasetConfig"""
        return product(self.experiment_types, self.num_letters, self.response_types, self.phases, self.electrodes)

    @property
    def md5(self):
        """md5 of this config; uselful as if of this particular specification of dataset"""
        return hashlib.md5(str(self).encode()).hexdigest()
