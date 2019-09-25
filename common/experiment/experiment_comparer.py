from collections import namedtuple
from enum import Enum, auto

import pandas as pd

from common.algorithms.algorithm import Algorithm
from common.dataset.dataset import Dataset
from common.experiment.experiment_analizer import Test, TestsAnalizer, summary_columns, summary_row, \
    layout_summary_columns

TestLayout = namedtuple('TestLayout', 'training_set test_set')
TestStrategy = namedtuple("TestStrategy", "key split_strategy")

def cross(dataset: Dataset, train_percentage):
    layouts = []
    randomized_ds = dataset.build_random_sample_dataset(1, replace=False)
    rounds = int(1/train_percentage)
    for i in range(0, rounds):
        layouts.append(TestLayout(*randomized_ds.partition_dataset(rounds, i)))
    return layouts


class TestType(Enum):
    FULL_TRAINING = TestStrategy(auto(), lambda dataset, train_percentage: [TestLayout(dataset, dataset)])
    # This means the training AND testing set will be the whole dataset, ignores train_percentage
    FULL_TEST = TestStrategy(auto(), lambda dataset, train_percentage: [TestLayout(dataset.build_random_sample_dataset(train_percentage), dataset)])
    # This means the test set will be the whole dataset, and the training set a subset of it
    DISJOINT = TestStrategy(auto(), lambda dataset, train_percentage: [TestLayout(*dataset.split_dataset(train_percentage))])
    # DISJOINT means the training and test set will be disjoint
    SAMPLED = TestStrategy(auto(), lambda dataset, train_percentage: [TestLayout(dataset.build_random_sample_dataset(train_percentage),
                                                         dataset.build_random_sample_dataset(1 - train_percentage))])
    # SAMPLED means the training and test set will draw n (total-n) samples from the full set. There may be repeated tests
    CROSS = TestStrategy(auto(), lambda dataset, train_percentage: cross(dataset, train_percentage))


    def split_dataset(self, dataset:Dataset, train_percentage, rounds):
        strategies = []
        for round in range(0, rounds):
            for strategy in self.value.split_strategy(dataset, train_percentage):
                strategies.append(strategy)
        return  strategies


class Comparer():
    def __init__(self, dataset: Dataset, train_percentage, algorithms: [Algorithm], rounds=1, test_type: TestType=TestType.FULL_TEST):
        self.algorithms = algorithms
        self.dataset = dataset
        self.test_type = test_type
        self.train_percentage = train_percentage
        self.tests_layouts = test_type.split_dataset(dataset, train_percentage, rounds)
        # self.training_set, self.test_set

        self.last_confusion = None
        self.results = self.calculate_results()
        self.result_summary = self.calculate_summary()

    def calculate_results(self):
        results = []
        for algorithm in self.algorithms:
            layout_result_analizers = []
            layout_result_metrics = []
            for test_layout in self.tests_layouts:
                result_analizer = self.calculate_layout_result(test_layout, algorithm)
                layout_result_analizers.append(result_analizer)
                layout_result_metrics.append(result_analizer.metrics)
            results.append(TestsAnalizer.calc_summary(pd.concat(layout_result_metrics), algorithm.get_tags(), len(self.tests_layouts)))
            self.last_confusion = layout_result_analizers[-1].confusion
        return results


    def calculate_layout_result(self, test_layout: TestLayout, algorithm):
        # print("Training " + str(algorithm.get_tags()["Algorithm"]) + " with " + str(len(cls.training_set.getRows())))
        algorithm.train(test_layout.training_set)
        tests = []
        for row in test_layout.test_set.getRows().itertuples(index=False):
            rowDict = row._asdict()
            guessed = algorithm.evaluate(rowDict)
            actual = rowDict[test_layout.training_set.getClassAttr()]
            # if guessed != actual:
            #     print("Missed!:" + str(row))
            tests.append(Test(actual, guessed))
        return TestsAnalizer(tests, test_layout.training_set.getClassAttrValues(), algorithm.get_tags())


    def calculate_summary(self):
        all_tags = []
        for algorithm in self.algorithms:
            all_tags = all_tags + list(algorithm.get_tags().keys())
        all_columns = all_tags + layout_summary_columns
        all_columns = list(dict.fromkeys(all_columns))
        summary = pd.DataFrame(data=0, index=range(0,len(self.algorithms)), columns=all_columns)

        for i in range(0, len(self.results)):
            for column in all_columns:
                if column in self.results[i].columns:
                    summary.loc[i, column] = self.results[i].loc[summary_row, column]
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
        strBuilder += str(len(self.tests_layouts))
        strBuilder += "\n\nSummary:\n"
        strBuilder += str(self.result_summary)
        return strBuilder