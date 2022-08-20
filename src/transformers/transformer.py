from abc import ABC, abstractmethod


class Transformer(ABC):
    @abstractmethod
    def transform(self):
        """transforms dataset's X and y into a different format, e.g. a different domain"""
