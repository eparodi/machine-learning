import pandas as pd

import common.dataset.dataset_builder as db
from common.algorithms.decision_tree import DecisionTree
from common.algorithms.random_forest import RandomForest
from common.experiment.experiment_comparer import Comparer, TestType
from common.utils.information import InfGainFunction
import matplotlib.pyplot as plt
import common.utils.strings as s


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

ds = db.create_titanic_dataset()
algorithms = []

algorithms.append(DecisionTree(max_nodes=1, inf_gain_function=InfGainFunction.SHANNON))
algorithms.append(DecisionTree(max_nodes=2, inf_gain_function=InfGainFunction.SHANNON))
algorithms.append(DecisionTree(max_nodes=3, inf_gain_function=InfGainFunction.SHANNON))
algorithms.append(DecisionTree(max_nodes=4, inf_gain_function=InfGainFunction.SHANNON))
algorithms.append(DecisionTree(max_nodes=5, inf_gain_function=InfGainFunction.SHANNON))
algorithms.append(DecisionTree(max_nodes=6, inf_gain_function=InfGainFunction.SHANNON))
algorithms.append(DecisionTree(max_nodes=7, inf_gain_function=InfGainFunction.SHANNON))
algorithms.append(DecisionTree(max_nodes=8, inf_gain_function=InfGainFunction.SHANNON))
algorithms.append(DecisionTree(max_nodes=9, inf_gain_function=InfGainFunction.SHANNON))
algorithms.append(DecisionTree(max_nodes=10, inf_gain_function=InfGainFunction.SHANNON))


comparer = Comparer(ds, 0.4, algorithms,rounds=10, test_type=TestType.DISJOINT)

print(comparer)

ax = plt.gca()
comparer.result_summary.plot(kind='line',x='MaxNodes',y=s.accuracy_mean,ax=ax)
plt.show()