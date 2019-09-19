import pandas as pd
from common import experiment_builder as exp
from common.dataset_manager import Dataset
from common.decision_tree import DecisionTree
from common.experiment_evaluator import Evaluator, Test
from common.naive_bayes import NaiveBayes

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

ds = Dataset.create_titanic_dataset()
class_col, class_values, attrs, rows = ds.getClassAttr(), ds.getClassAttrValues(), ds.getAttributes(), ds.getRows()
training, test = exp.random_with_replacement_split(0.5, rows)
print(training)
print(test)

tree = DecisionTree(training, attrs, class_col)
print("\n")
print(tree.get_tree())

tests = []
for row in test.itertuples(index=False):
    rowDict = row._asdict()
    guessed = tree.evaluate(rowDict)
    actual = rowDict[class_col]
    tests.append(Test(actual, guessed))

print(Evaluator(tests, class_values))
