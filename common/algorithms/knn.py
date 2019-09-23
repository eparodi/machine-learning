from .algorithm import Algorithm
from common.dataset.dataset import Dataset

import common.utils.strings as s
import numpy as np

class KNN(Algorithm):

    DISTANCE = 'dist'
    attributes = []
    classAttribute = None

    def __init__(self, k=5, weighted=False):
        super().__init__()
        self.weighted = weighted
        self.k = k

    def train(self, dataset: Dataset):
        self.attributes = dataset.getAttributes()
        self.classAttribute = dataset.getClassAttr()
        self.rows = dataset.getRows()

    def __weighted_evaluation(self, weights):
        weights[self.DISTANCE] = 1 / weights[self.DISTANCE]
        sums = weights.head(self.k).groupby(self.classAttribute).sum().sort_values(self.DISTANCE)
        return sums.idxmax()[self.DISTANCE]

    def __unweighted_evaluation(self, weights):
        result = None
        k = self.k
        while not result:
            sums = weights.head(k).groupby(
                self.classAttribute).count().sort_values(self.DISTANCE, ascending=False)
            values = list(sums.values[:, 0])
            if values.count(max(values)) == 1:
                return sums.idxmax()[self.DISTANCE]
            k += 1

    def evaluate(self, values_dict):
        distances = self.rows
        distances[self.DISTANCE] = 0
        for attr in self.attributes:
            distances[self.DISTANCE] += (distances[attr] -
                                         values_dict[attr]) ** 2
        distances[self.DISTANCE] = np.sqrt(distances[self.DISTANCE])
        distances = distances[[self.DISTANCE, self.classAttribute]]
        distances = distances.sort_values(by=[self.DISTANCE])
        if self.weighted:
            return self.__weighted_evaluation(distances)
        else:
            return self.__unweighted_evaluation(distances)

    def get_tags(self):
        return {
            s.algorithm: "KNN",
            "weighted": self.weighted,
            "k": self.k,
        }
