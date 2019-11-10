from .algorithm import Algorithm
from common.dataset.dataset import Dataset

from sklearn import linear_model
from sklearn.model_selection import train_test_split

import common.utils.strings as s
import numpy

class LogisticalRegression(Algorithm):

    def __init__(self, *args, **kwargs):
        self.lm = linear_model.LogisticRegression(**kwargs)

    def train(self, dataset: Dataset):
        self.classAttribute = dataset.getClassAttr()
        rows = dataset.getRows()
        self.X_train = rows.drop(self.classAttribute, axis=1)
        self.y_train = rows[self.classAttribute]
        self.lm.fit(self.X_train, self.y_train)

    def evaluate(self, values_dict):
        array = []
        for key, value in values_dict.items():
            if key != self.classAttribute:
                array.append(value)
        nparray = numpy.array(array).reshape(1, -1)
        return self.lm.predict(nparray)

    def get_tags(self):
        return {
            s.algorithm: "Logistic Regression",
        }

