from .algorithm import Algorithm
from common.dataset.dataset import Dataset

import common.utils.strings as s
import numpy as np    

def complete_link(data1, data2):
    max_distance = 0
    for v1 in data1:
        for v2 in data2:
            distance = 0
            for key in v1.keys():
                distance += (v1[key] - v2[key]) ** 2
            if max_distance < distance:
                max_distance = distance 
    return max_distance

def single_link(data1, data2):
    min_distance = np.inf
    for v1 in data1:
        for v2 in data2:
            distance = 0
            for key in v1.keys():
                distance += (v1[key] - v2[key]) ** 2
            if min_distance > distance:
                min_distance = distance
    return min_distance

def mean_distance(data1, data2):
    mean_distance = 0
    number = 0
    for v1 in data1:
        for v2 in data2:
            distance = 0
            for key in v1.keys():
                distance += (v1[key] - v2[key]) ** 2
            mean_distance += distance
            number += 1
    return mean_distance / number

def get_centroid_from_data(data):
    centroid = {key: 0 for key in data[0].keys()}
    for d in data:
        for key in centroid:
            centroid[key] += d[key]
    for key in centroid:
        centroid[key] /= len(data)
    return centroid

def centroid_distance(data1, data2):
    centroid1 = get_centroid_from_data(data1)
    centroid2 = get_centroid_from_data(data2)
    distance = 0
    for key in centroid1.keys():
        distance += (centroid1[key] - centroid2[key]) ** 2
    return distance

class HierarchicalClustering(Algorithm):

    COMPLETE_LINK = "complete_link"
    SINGLE_LINK = "single_link"
    MEAN = "mean"
    CENTROID = "centroid"
    distances = {
        COMPLETE_LINK : complete_link,
        SINGLE_LINK: single_link,
        MEAN: mean_distance,
        CENTROID: centroid_distance,
    }

    def __init__(self, method):
        super().__init__()
        self.order = 0
        self.method = method

    def train(self, dataset: Dataset):
        self.dataset = dataset
        self.groups = []
        rows = dataset.getRows()
        self.create_initial_groups(rows)
        while len(self.groups) != 1:
            self.reassign()

    def create_initial_groups(self, rows):
        for row in rows.iterrows():
            self.groups.append({
                "previous": None,
                "order": self.order,
                "data": [row[1]]
            })

    def reassign(self):
        size = len(self.groups)
        matrix = np.matrix(np.zeros((size, size)))
        for i in range(0, size):
            for j in range(i, size):
                if i == j:
                    matrix[i, j] = np.inf
                else:
                    matrix[i, j] = self.calculate_distance(i, j)
                    matrix[j, i] = self.calculate_distance(i, j)
        min_val = matrix.min()
        x, y = np.where(matrix == min_val)
        self.order += 1
        x_group = self.groups[x[0]]
        y_group = self.groups[y[0]]
        group = {
            "previous": [x_group, y_group],
            "order": self.order,
            "data": x_group["data"] + y_group["data"]
        }
        index = [x[0], y[0]]
        index.sort(reverse=True)
        del self.groups[index[0]]
        del self.groups[index[1]]
        self.groups.append(group)

    def remove_nonattributes(self, data):
        new_data = []
        for d in data:
            new_data.append({key: d[key]
                             for key in self.dataset.attributes})
        return new_data

    def calculate_distance(self, i, j):
        data1 = self.remove_nonattributes(self.groups[i]["data"])
        data2 = self.remove_nonattributes(self.groups[j]["data"])
        return self.distances[self.method](data1, data2)

    def get_groups(self, n):
        groups = self.groups
        i = 1
        while i != n:
            order = 0
            next_group = None
            for group in groups:
                if group["order"] > order:
                    next_group = group
                    order = group["order"]
            groups = [group for group in groups if group["order"] != order]
            groups += next_group["previous"]
            i += 1
        return groups

    def evaluate(self):
        raise Exception("Not valid!")

    def get_tags(self):
        return {
            s.algorithm: "HierarchicalClustering",
            "distance": self.method
        }
