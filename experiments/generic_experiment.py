import pandas as pd

import common.dataset.dataset_builder as db
from common.algorithms.decision_tree import DecisionTree
from common.experiment.experiment_comparer import Comparer, TestType
from common.utils.information import InfGainFunction

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

ds = db.create_titanic_dataset()
algorithms = []
# algorithms.append(DecisionTree(used_percentage=0.1))
# algorithms.append(DecisionTree(used_percentage=0.25))
algorithms.append(DecisionTree(max_nodes=2, inf_gain_function=InfGainFunction.ERROR))
algorithms.append(DecisionTree(max_nodes=2, inf_gain_function=InfGainFunction.GINI))
algorithms.append(DecisionTree(max_nodes=2, inf_gain_function=InfGainFunction.SHANNON))
algorithms.append(DecisionTree(max_nodes=4, inf_gain_function=InfGainFunction.ERROR))
algorithms.append(DecisionTree(max_nodes=4, inf_gain_function=InfGainFunction.GINI))
algorithms.append(DecisionTree(max_nodes=4, inf_gain_function=InfGainFunction.SHANNON))
algorithms.append(DecisionTree(max_nodes=6, inf_gain_function=InfGainFunction.ERROR))
algorithms.append(DecisionTree(max_nodes=6, inf_gain_function=InfGainFunction.GINI))
algorithms.append(DecisionTree(max_nodes=6, inf_gain_function=InfGainFunction.SHANNON))
# algorithms.append(DecisionTree(max_nodes=8))
# algorithms.append(Lottery(weighted=True))
# algorithms.append(Lottery(weighted=False))

comparer = Comparer(ds, 0.5, algorithms, test_type=TestType.DISJOINT)

print(comparer)
