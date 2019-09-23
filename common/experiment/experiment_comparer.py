from enum import Enum, auto

import pandas as pd

from common.algorithms.algorithm import Algorithm
from common.dataset.dataset import Dataset
from common.experiment.experiment_analizer import Test, TestsAnalizer, summary_columns, summary_row
from common.experiment.experiment_builder import random_with_replacement_split

class TestType(Enum):
    FULL_TRAINING = auto()
    # This means the training AND testing set will be the whole dataset, ignores train_percentage
    FULL_TEST = auto()
    # This means the test set will be the whole dataset, and the training set a subset of it
    DISJOINT = auto()
    # DISJOINT means the training and test set will be disjoint
    SAMPLED = auto()
    # SAMPLED means the training and test set will draw n (total-n) samples from the full set. There may be repeated tests

class Comparer():
    def __init__(self, dataset: Dataset, train_percentage, algorithms: [Algorithm], test_type: TestType=TestType.FULL_TEST):
        self.algorithms = algorithms
        self.dataset = dataset
        self.test_type = test_type
        self.train_percentage = train_percentage
        self.training_set, self.test_set = self.split_dataset()

        self.results = self.calculate_results()
        self.result_summary = self.calculate_summary()

    def split_dataset(self):
        if self.test_type == TestType.FULL_TEST:
            return self.dataset.build_random_sample_dataset(self.train_percentage), self.dataset
        elif self.test_type == TestType.FULL_TRAINING:
            return self.dataset, self.dataset
        elif self.test_type == TestType.DISJOINT:
            return self.dataset.split_dataset(self.train_percentage)
        elif self.test_type == TestType.SAMPLED:
            return self.dataset.build_random_sample_dataset(self.train_percentage),\
                   self.dataset.build_random_sample_dataset(1 - self.train_percentage)

    def calculate_results(cls):
        results = []
        for algorithm in cls.algorithms:
            print("Training " + str(algorithm.get_tags()["Algorithm"]) + " with " + str(len(cls.training_set.getRows())))
            algorithm.train(cls.training_set)
            tests = []
            for row in cls.test_set.getRows().itertuples(index=False):
                rowDict = row._asdict()
                guessed = algorithm.evaluate(rowDict)
                actual = rowDict[cls.training_set.getClassAttr()]
                # if guessed != actual:
                #     print("Missed!:" + str(row))
                tests.append(Test(actual, guessed))
            results.append(TestsAnalizer(tests, cls.training_set.getClassAttrValues(), algorithm.get_tags()))
        return results

    def calculate_summary(self):
        all_tags = []
        for algorithm in self.algorithms:
            all_tags = all_tags + list(algorithm.get_tags().keys())
        all_columns = all_tags + summary_columns
        all_columns = list(dict.fromkeys(all_columns))
        summary = pd.DataFrame(data=0, index=range(0,len(self.algorithms)), columns=all_columns)

        for i in range(0, len(self.results)):
            for column in all_columns:
                if column in self.results[i].summary.columns:
                    summary.loc[i, column] = self.results[i].summary.loc[summary_row, column]
                else:
                    summary.loc[i, column] = "-"
        return summary

    def __str__(self):
        strBuilder = ""
        strBuilder += "\nTest type:"
        strBuilder += str(self.test_type.name)
        strBuilder += "\nTrain Percentage:"
        strBuilder += str(self.train_percentage)
        strBuilder += "\nData Set Size:"
        strBuilder += str(len(self.dataset.getRows()))
        strBuilder += "\nTraining Set Size:"
        strBuilder += str(len(self.training_set.getRows()))
        strBuilder += "\nTest Set Size:"
        strBuilder += str(len(self.test_set.getRows()))
        strBuilder += "\n\nSummary:\n"
        strBuilder += str(self.result_summary)
        return strBuilder