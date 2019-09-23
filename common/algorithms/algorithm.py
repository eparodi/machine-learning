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


# algorithms = build_algorithm_permutation(DecisionTree, [{"max_nodes":range(0,4)}])

# def build_algorithm_permutation(algorithm: Algorithm, parameters: [()]):
#     algorithms = []
#
#     for param in parameters:
#         algorithms.append(algorithm(**param))
#     return algorithms
