from abc import ABC, abstractmethod

from common.dataset.dataset import Dataset


class Algorithm(ABC):
    def __init__(self):
        self.trained = False

    @abstractmethod
    def evaluate(self, values_dict: dict):
        pass

    @abstractmethod
    def get_tags(self):
        pass

    @abstractmethod
    def train(self, dataset: Dataset):
        pass

    def is_trained(self):
        return self.is_trained