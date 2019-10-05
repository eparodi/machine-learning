from .algorithm import Algorithm
from common.dataset.dataset import Dataset
from sklearn.svm import SVC

import common.utils.strings as s

class SVM(Algorithm):

    def __init__(self, kernel="linear"):
        self.kernel = kernel
        self.svclassifier = SVC(kernel=kernel)

    def train(self, dataset: Dataset):
        classAttribute = dataset.getClassAttr()
        rows = dataset.getRows()
        self.X_train = rows.drop(classAttribute, axis=1)
        self.y_train = rows[classAttribute]
        self.svclassifier.fit(self.X_train, self.y_train)

    def evaluate(self, values_dict):
        array = []
        del values_dict["_2"]
        for value in values_dict.values():
            array.append(value)
        return self.svclassifier.predict([array])

    def get_tags(self):
        return {
            s.algorithm: "SVM",
            "kernel": self.kernel,
        }

