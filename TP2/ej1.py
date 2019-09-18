import pandas as pd
from common.dataset_manager import Dataset
from common.decision_tree import DecisionTree

from common.information import inf_gain
from common.tree import Tree

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

ds = Dataset.create_britons_dataset()
class_col, allCategories, sports = ds.getClassAttr(), ds.getAttributes(), ds.getRows()

print(class_col)
print(allCategories)
print(sports)

tree = DecisionTree.from_dataset(ds)
print(tree.get_tree())

