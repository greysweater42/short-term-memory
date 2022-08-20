from pathlib import Path
from typing import List, Tuple
import pickle

import config
from src.observation import Observation

from multiprocessing import Pool
from src.utils import timeit
from .dataset_config import DatasetConfig


class DatasetLoader:
    """reads data for a given DatasetConfig; for efficiency stores data in cache for a given config"""
    def __init__(self, config: DatasetConfig, from_cache: bool = True) -> None:
        self.from_cache = from_cache
        self.config = config

        self.X: List[Observation] = []

    @timeit
    def load(self):
        self.X: List[Observation] = []  # makes this method idempotent
        cache_filenames = [filename.stem for filename in config.DATASET_CACHE.iterdir()]
        if self.config.md5 in cache_filenames and self.from_cache:
            with open(config.DATASET_CACHE / self.config.md5, "rb") as file_:
                self.X = pickle.load(file_)
        else:
            self._load_observations()
            with open(config.DATASET_CACHE / self.config.md5, "wb") as file_:
                pickle.dump(self.X, file_)
        return self.X

    @timeit
    def _load_observations(self) -> None:
        self.X: List[Observation] = []  # makes this method idempotent

        with Pool(config.CPUs) as executor:
            combinations = list(self.config.combinations)
            results = executor.imap(self._load_observations_for_combination, combinations)
            for result in results:
                self.X += result

    @staticmethod
    def _load_observations_for_combination(combination: Tuple) -> List[Observation]:
        observation_path = Observation.path_for_searching(
            experiment_type=combination[0],
            num_letters=combination[1],
            response_type=combination[2],
            name=combination[3],
            electrode=combination[4]
        )
        # observation_path can be sth like "data_cache/*/5/M/correct/*/delay/T6.feather"
        paths = Path().rglob(str(observation_path))
        return [Observation.from_path(path) for path in paths]
