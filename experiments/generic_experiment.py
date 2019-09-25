import pandas as pd

import common.dataset.dataset_builder as db
from common.algorithms.decision_tree import DecisionTree
from common.algorithms.lottery import Lottery
from common.algorithms.random_forest import RandomForest
from common.experiment.experiment_comparer import Comparer, TestType

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

ds = db.create_titanic_dataset()
algorithms = []
algorithms.append(RandomForest(n_trees=3))
algorithms.append(RandomForest(n_trees=5))
algorithms.append(RandomForest(n_trees=7))
algorithms.append(DecisionTree(max_nodes=3))
algorithms.append(DecisionTree(max_nodes=5))
algorithms.append(DecisionTree(max_nodes=7))
algorithms.append(Lottery(weighted=True))
algorithms.append(Lottery(weighted=False))

comparer = Comparer(ds, 0.2, algorithms, test_type=TestType.FULL_TEST)

print(comparer)
# print(comparer.last_confusion)
