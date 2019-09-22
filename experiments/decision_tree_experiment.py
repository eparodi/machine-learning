import pandas as pd
import common.dataset.dataset_builder as db

from common.algorithms.decision_tree import DecisionTree
from common.dataset.dataset import Dataset
from common.experiment import experiment_builder as exp
from common.experiment.experiment_analizer import TestsAnalizer, Test

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

ds = db.create_rio_dataset()
training, test = ds, ds
print(training)
print(test)

tree = DecisionTree()
tree.train(training)
print("\n")
print(tree.get_tree())

tests = []
for row in test.getRows().itertuples(index=False):
    rowDict = row._asdict()
    guessed = tree.evaluate(rowDict)
    actual = rowDict[ds.getClassAttr()]
    tests.append(Test(actual, guessed))

print(TestsAnalizer(tests, ds.getClassAttrValues()))
