from collections import namedtuple

import pandas as pd

Test = namedtuple('Test', 'actual guessed')


class Evaluator:

    def __init__(self, tests: [Test], class_attr_values):
        hit = 0
        miss = 0
        total = 0

        confusion = pd.DataFrame(data=0, index=class_attr_values, columns=class_attr_values)
        metrics = pd.DataFrame(data=0, index=class_attr_values,
                               columns=["Accuracy", "Precision", "Recall", "F1-score", "TruePositives", "TrueNegatives",
                                        "FalsePositives", "FalseNegatives", "Total"])

        for test in tests:
            guessed = test.guessed
            actual = test.actual
            total += 1
            try:
                oldValue = confusion.loc[actual, guessed]
                confusion.loc[actual, guessed] = int(oldValue) + 1
            except KeyError:
                print("ERROR!")

            Evaluator.addOneToDataframe(metrics, actual, "Total", metrics)
            if (guessed == actual):
                hit += 1
            else:
                miss += 1

        confusion.loc['Column_Total'] = confusion.sum(numeric_only=True, axis=0)
        confusion.loc[:, 'Row_Total'] = confusion.sum(numeric_only=True, axis=1)

        for category in class_attr_values:
            metrics.loc[category, "FalsePositives"] = confusion.loc["Column_Total", category] - confusion.loc[
                category, category]
            metrics.loc[category, "TruePositives"] = confusion.loc[category, category]
            metrics.loc[category, "FalseNegatives"] = confusion.loc[category, "Row_Total"] - confusion.loc[
                category, category]
            metrics.loc[category, "TrueNegatives"] = confusion.loc["Column_Total", "Row_Total"] + confusion.loc[
                category, category] - confusion.loc["Column_Total", category] - confusion.loc[category, "Row_Total"]

        metrics["Accuracy"] = (metrics["TruePositives"] + metrics["TrueNegatives"]) / confusion.loc[
            "Column_Total", "Row_Total"]
        metrics["Precision"] = metrics["TruePositives"] / (metrics["TruePositives"] + metrics["FalsePositives"])
        metrics["Recall"] = metrics["TruePositives"] / (metrics["TruePositives"] + metrics["FalseNegatives"])
        metrics["F1-score"] = (2 * metrics["Precision"] * metrics["Recall"]) / (
        metrics["Precision"] + metrics["Recall"])

        self.metrics = metrics
        self.confusion = confusion
        self.correct = hit/total

    @staticmethod
    def addOneToDataframe(dataframe, category, metric, metrics):
        dataframe.loc[category, metric] = int(metrics.loc[category, metric]) + 1
