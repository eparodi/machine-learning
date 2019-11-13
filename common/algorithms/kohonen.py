import random

import pandas as pd

from .algorithm import Algorithm
from common.dataset.dataset import Dataset

import common.utils.strings as s
import numpy as np

class Kohonen(Algorithm):

    nodes = []
    NODE_X = "NODE_X"
    NODE_Y = "NODE_Y"
    NODE_CLASS = "NODE_CLASS"
    NODE_CLOSEST = "NODE_CLOSEST"

    def __init__(self, n=5, rounds=1, distance_weight=0.1, iteration_weight=0.1):
        super().__init__()
        self.n = n
        self.rounds = rounds
        self.distance_weight = distance_weight
        self.iteration_weight = iteration_weight

    def train(self, dataset: Dataset):
        self.attributes = dataset.getAttributes()
        self.classAttribute = dataset.getClassAttr()
        rows = dataset.getRows()
        self.nodes = []

        for x in range(0, self.n):
            node_row = []
            for y in range(0, self.n):
                node = {}
                node[self.NODE_X] = x
                node[self.NODE_Y] = y
                for attr in dataset.getAttributes():
                    node[attr] = random.uniform(0, 1)
                node_row.append(node)
            self.nodes.append(node_row)

        iteration = 0
        for round in range(0, self.rounds):
            shuffled_rows = rows.sample(frac=1)
            for row in shuffled_rows.itertuples(index=False):
                rowDict = row._asdict()
                closest_node = self.closest_node_weight(rowDict)
                neighbours_with_distance = self.get_nodes_closest_than(1.2, closest_node)
                for neighbor in neighbours_with_distance:
                    self.update_node_weight(neighbor, iteration, rowDict)
                iteration += 1

        for x in range(0, self.n):
            for y in range(0, self.n):
                node = self.nodes[x][y]
                closest_class = self.closest_class_to_node(node, rows)
                node[self.NODE_CLASS] = closest_class

    def evaluate(self, values_dict):
        return self.closest_node_weight(values_dict)[self.NODE_CLASS]

    def get_tags(self):
        return {
            s.algorithm: "KOHONEN",
            "n": self.n,
            "rounds": self.rounds,
            "dist_w": self.distance_weight,
            "iter_w": self.iteration_weight
        }

    def update_node_weight(self, node_with_distance, iteration, rowDict):
        node, distance = node_with_distance
        for attr in self.attributes:
            value_diff = rowDict[attr] - node[attr]
            weight_diff = self.distance_factor(distance) * self.iteration_factor(iteration) * value_diff
            node[attr] = node[attr] + weight_diff

    def closest_node_weight(self, row):
        closest = None
        closest_distance = float("inf")
        for x in range(0, self.n):
            for y in range(0, self.n):
                node = self.nodes[x][y]
                distance = 0
                for attr in self.attributes:
                    distance += (node[attr] - row[attr]) ** 2
                if distance <= closest_distance:
                    closest = node
                    closest_distance = distance
        return closest

    def closest_class_to_node(self, node, rows):
        closest_class = None
        closest_distance = float("inf")
        for row in rows.itertuples(index=False):
            rowDict = row._asdict()
            distance = 0
            for attr in self.attributes:
                distance += (node[attr] - rowDict[attr]) ** 2
            if distance <= closest_distance:
                closest_class = rowDict[self.classAttribute]
                closest_distance = distance
        return closest_class

    def distance_factor(self, distance):
        if distance == 0:
            factor = 2
        else:
            factor = 1/(distance**2)

        factor = factor * self.distance_weight / 2
        if factor < 0:
            return 0
        else:
            return factor

    def iteration_factor(self, iteration):
        factor = (1 - iteration/(self.n * self.n * self.rounds))
        if factor < 0:
            return 0
        else:
            return factor * self.iteration_weight

    def distance_function(self, current_node, other_node):
        eucl_sq_distance = (current_node[self.NODE_X] - other_node[self.NODE_X])**2 + (current_node[self.NODE_Y] - other_node[self.NODE_Y])**2
        return round(eucl_sq_distance ** 0.5)

    def get_nodes_closest_than(self, max_distance, node):
        neighbours = []
        for x in range(0, len(self.nodes)):
            for y in range(0, len(self.nodes[x])):
                dist = self.distance_function(node, self.nodes[x][y])
                if dist < max_distance:
                    neighbours.append((self.nodes[x][y], dist))
        return neighbours

    def get_nodes(self):
        return self.nodes

    def get_nodes_df(self):
        flat_nodes = [item for sublist in self.nodes for item in sublist]
        nodes_df = pd.DataFrame(flat_nodes, columns=["NODE_X", "NODE_Y", "NODE_CLASS"] + self.attributes)
        return nodes_df


