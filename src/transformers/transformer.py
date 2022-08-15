from abc import ABC, abstractmethod


class Transformer(ABC):
    def __init__(self) -> None:
        """generic init"""

    @abstractmethod
    def transform(self):
        """transforms eeg pd.Series or pd.DataFrame to an easier to a different format, e.g. a different domain"""
