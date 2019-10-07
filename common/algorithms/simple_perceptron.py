import random

from common.algorithms.algorithm import Algorithm
from common.dataset.dataset import Dataset



class SimplePerceptron(Algorithm):
    def __init__(self, learning_rate=0.2, epochs=1):
        super().__init__()
        self.weight = []
        self.attrOrder = {}
        self.clazz_attr_values = {}
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activate(self, result):
        return 0 if result < self.weight[0] else 1

    def evaluate(self, values_dict: dict):
        return self.activate(self.sum_weights(values_dict))

    def sum_weights(self, values_dict: dict):
        result = 0
        for key in self.attrOrder.keys():
            result += self.weight[self.attrOrder[key]] * values_dict[key]
        return result

    def get_tags(self):
        return {"Algorithm": "SimplePerceptron",
                "Epochs": self.epochs,
                "LearningRate": self.learning_rate,
                }

    def train(self, dataset: Dataset):
        self.weight.clear()
        self.weight.append(random.uniform(0.1, 0.9))
        for attr in dataset.getAttributes():
            self.attrOrder[attr] = len(self.weight)
            self.weight.append(random.uniform(0.1, 0.9))

        for epoch in range(0, self.epochs):
            for row in dataset.getRows().itertuples():
                values = row._asdict()
                res = self.sum_weights(values)
                for attr in dataset.getAttributes():
                    delta = self.learning_rate * (values[dataset.getClassAttr()] - self.activate(res)) * values[attr]
                    self.weight[self.attrOrder[attr]] += delta

        return

    def __str__(self):
        strBuilder = ""
        strBuilder += "\nOrder:\n"
        strBuilder += str(self.attrOrder)
        strBuilder += "\n\nWeights:\n"
        strBuilder += str(self.weight)
        return strBuilder