import pandas as pd

import common.dataset.dataset_builder as db
from common.algorithms.decision_tree import DecisionTree
from common.algorithms.random_forest import RandomForest
from common.experiment.experiment_comparer import Comparer, TestType
from common.utils.information import InfGainFunction

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

ds = db.create_titanic_dataset()
algorithms = []
# b,c)
# algorithms.append(DecisionTree(max_nodes=7, inf_gain_function=InfGainFunction.SHANNON))
# algorithms.append(DecisionTree(max_nodes=7, inf_gain_function=InfGainFunction.GINI))
# algorithms.append(DecisionTree(max_nodes=7, inf_gain_function=InfGainFunction.ERROR))
# d)
algorithms.append(RandomForest(inf_gain_function=InfGainFunction.SHANNON))
algorithms.append(RandomForest(inf_gain_function=InfGainFunction.GINI))
algorithms.append(RandomForest(inf_gain_function=InfGainFunction.ERROR))

# algorithms.append(DecisionTree(max_nodes=2, inf_gain_function=InfGainFunction.ERROR))
# algorithms.append(DecisionTree(max_nodes=2, inf_gain_function=InfGainFunction.GINI))
# algorithms.append(DecisionTree(max_nodes=2, inf_gain_function=InfGainFunction.SHANNON))
# algorithms.append(DecisionTree(max_nodes=4, inf_gain_function=InfGainFunction.ERROR))
# algorithms.append(DecisionTree(max_nodes=4, inf_gain_function=InfGainFunction.GINI))
# algorithms.append(DecisionTree(max_nodes=4, inf_gain_function=InfGainFunction.SHANNON))
# algorithms.append(DecisionTree(max_nodes=6, inf_gain_function=InfGainFunction.ERROR))
# algorithms.append(DecisionTree(max_nodes=6, inf_gain_function=InfGainFunction.GINI))
# algorithms.append(DecisionTree(max_nodes=6, inf_gain_function=InfGainFunction.SHANNON))

comparer = Comparer(ds, 0.5, algorithms,rounds=5, test_type=TestType.FULL_TEST)

print(comparer)
print(comparer.last_confusion)