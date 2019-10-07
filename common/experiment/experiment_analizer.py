from collections import namedtuple
from enum import auto, Enum

import pandas as pd
import common.utils.strings as s

Test = namedtuple('Test', 'actual guessed')
metrics_columns = ["Accuracy", "Precision", "Recall", "F1-score", "TruePositives", "TrueNegatives", "FalsePositives", "FalseNegatives", "Weight", "Total"]
summary_columns = [s.correct, s.accuracy_mean, s.accuracy_std, s.f1_mean, s.f1_std]
layout_summary_columns = [s.rounds, s.accuracy_mean, s.f1_mean]
summary_row = "0"


class TestsAnalizer:
    def __init__(self, tests: [Test], class_attr_values, tags={}):
        hit = 0
        miss = 0
        total = 0

        confusion = pd.DataFrame(data=0, index=class_attr_values, columns=class_attr_values)
        metrics = pd.DataFrame(data=0, index=class_attr_values, columns=metrics_columns)

        for test in tests:
            guessed = test.guessed
            actual = test.actual
            total += 1
            try:
                oldValue = confusion.loc[actual, guessed]
                confusion.loc[actual, guessed] = int(oldValue) + 1
            except KeyError:
                print(self.stack_trace(confusion, actual, guessed))

            TestsAnalizer.addOneToDataframe(metrics, actual, "Total", metrics)
            if guessed == actual:
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
        metrics["F1-score"] = (2 * metrics["Precision"] * metrics["Recall"]) / (metrics["Precision"] + metrics["Recall"])
        metrics["Weight"] = metrics["Total"]/confusion.loc["Column_Total", "Row_Total"]

        tag_keys = list(tags)
        summary = pd.DataFrame(data=0, index=["0"], columns=tag_keys + summary_columns)

        summary.loc[summary_row, s.accuracy_mean] = metrics[s.accuracy].mean()
        summary.loc[summary_row, s.accuracy_std] = metrics[s.accuracy].std()
        summary.loc[summary_row, s.f1_mean] = metrics[s.f1].mean()
        summary.loc[summary_row, s.f1_std] = metrics[s.f1].std()
        summary.loc[summary_row, s.correct] = hit/total

        for tag_key in tag_keys:
            summary.loc[summary_row, tag_key] = tags[tag_key]

        self.metrics = metrics
        self.confusion = confusion
        self.summary = summary

    @staticmethod
    def calc_summary(metrics, tags, size):
        tag_keys = list(tags)
        summary = pd.DataFrame(data=0, index=["0"], columns=tag_keys + layout_summary_columns)

        summary.loc[summary_row, s.accuracy_mean] = (metrics[s.accuracy]*metrics[s.weight]).sum() / size
        summary.loc[summary_row, s.f1_mean] = (metrics[s.f1]*metrics[s.weight]).sum() / size
        summary.loc[summary_row, s.rounds] = size

        for tag_key in tag_keys:
            summary.loc[summary_row, tag_key] = tags[tag_key]

        return summary

    @staticmethod
    def addOneToDataframe(dataframe, category, metric, metrics):
        dataframe.loc[category, metric] = int(metrics.loc[category, metric]) + 1

    def __str__(self):
        strBuilder = ""
        strBuilder += "\nConfusion Matrix:\n"
        strBuilder += str(self.confusion)
        strBuilder += "\n\nMetrics by Category:\n"
        strBuilder += str(self.metrics)
        strBuilder += "\n\nSummary:\n"
        strBuilder += str(self.summary)
        return strBuilder

    def stack_trace(self, confusion, actual, guessed):
        strBuilder = "StackTrace:"
        strBuilder += "\nConfusion:\n"
        strBuilder += str(confusion)
        strBuilder += "\nActual:\n"
        strBuilder += str(actual)
        strBuilder += "\nGuessed:\n"
        strBuilder += str(guessed)
        return strBuilder
