import pandas as pd

import common.dataset.dataset_builder as db
from common.algorithms.decision_tree import DecisionTree
from common.algorithms.lottery import Lottery
from common.experiment.experiment_comparer import Comparer, TestType

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

ds = db.create_titanic_dataset()
algorithms = []
algorithms.append(DecisionTree(used_percentage=0.1))
# algorithms.append(DecisionTree(used_percentage=0.25))
# algorithms.append(DecisionTree(used_percentage=0.5))
algorithms.append(DecisionTree(used_percentage=1))
algorithms.append(Lottery(weighted=True))
algorithms.append(Lottery(weighted=False))

comparer = Comparer(ds, 1, algorithms, test_type=TestType.FULL_TRAINING)

print(comparer)
