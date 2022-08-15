from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def train(self):
        """generic for training the model"""
