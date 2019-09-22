
import common.utils.strings as s
from common.algorithms.algorithm import Algorithm
from common.dataset.dataset import Dataset
import random

class Lottery(Algorithm):
    def __init__(self, weighted:bool=False):
        self.weighted = weighted

    def train(self, dataset: Dataset):
        self.clazz_attr_values = dataset.clazz_attr_values
        self.freq = dataset.getRows()[dataset.getClassAttr()].value_counts()
        self.total = self.freq.sum()

    def get_tags(self):
        return {s.algorithm: "Lottery",
                "Weighted": self.weighted}

    def evaluate(self, values_dict: dict):
        idx = random.randrange(0, self.total)
        limit = 0
        if self.weighted:
            for clazz_attr_value in self.clazz_attr_values:
                limit += self.freq[clazz_attr_value]
                if idx < limit:
                    return clazz_attr_value
            print("ERROR: weighed lottery random not working")
        return random.choice(self.clazz_attr_values)