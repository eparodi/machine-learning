import random

from random import sample

from common.algorithms.algorithm import Algorithm
from common.algorithms.decision_tree import DecisionTree
from common.dataset.dataset import Dataset
from common.utils.information import InfGainFunction

class KNearest(Algorithm):

    CLASS_KEY = 'assigned_class'

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.centroids = {klass: None for klass in range(0, n_classes)}

    def train(self, dataset: Dataset):
        classAttribute = dataset.getClassAttr()
        self.dataset = dataset
        self.columns = dataset.attributes
        self.data_frame = dataset.getRows().copy()
        self.assign_init_classes()
        changes = True
        while changes:
            self.calculate_centroids()
            changes = self.assign_with_centroids()

    def assign_init_classes(self):
        self.data_frame[self.CLASS_KEY] = None
        self.data_frame[self.CLASS_KEY] = self.data_frame[self.CLASS_KEY].apply(
            lambda r: random.randint(0, self.n_classes - 1),
        )

    def assign_with_centroids(self):
        changed = False
        series = []
        for row in self.data_frame.iterrows():
            klass = self.closest_centroid(row[1])
            series.append(klass)
            if klass != row[1][self.CLASS_KEY]:
                changed = True
        self.data_frame[self.CLASS_KEY] = series
        return changed

    def closest_centroid(self, row):
        closest = None
        closest_distance = float("inf")
        for klass, value in self.centroids.items():
            weights = [row[key] for key in self.columns]
            distance = 0
            for i in range(0, len(weights)):
                distance += (weights[i] - value[i]) ** 2
            if distance <= closest_distance:
                closest = klass
                closest_distance = distance
        return closest

    def calculate_centroids(self):
        count = {klass: 0 for klass in range(0, self.n_classes)}
        for key in self.centroids.keys():
            self.centroids[key] = [0 for x in range(0, len(self.columns))]
        
        for row in self.data_frame.iterrows():
            row_dict = row[1]
            count[row_dict[self.CLASS_KEY]] += 1
            i = 0
            for col in self.columns:
                self.centroids[row_dict[self.CLASS_KEY]][i] += row_dict[col]
                i += 1
        
        for key in self.centroids.keys():
            for i in range(0, len(self.columns)):
                self.centroids[key][i] /= count[key] if count[key] != 0 else float("inf")

    def get_tags(self):
        return {
            "Algorithm": "K-nearest",
            "Number of Classes": self.n_classes,
        }

    def evaluate(self, values_dict: dict):
        return self.closest_centroid(values_dict)

