import os, sys; sys.path.append(os.getcwd())

import pandas as pd

import common.dataset.dataset_builder as db
from common.algorithms.logistical_regression import LogisticalRegression
from common.algorithms.knn import KNN
from common.algorithms.knearest import KNearest
from common.experiment.experiment_comparer import Comparer, TestType

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

print("Without sex.")

ds = db.create_heart_dataset(["sex"])
lr = LogisticalRegression(solver='lbfgs')
algorithms = []
algorithms.append(lr)
algorithms.append(KNN())
algorithms.append(KNN(weighted=True))
algorithms.append(KNearest(2))

comparer = Comparer(ds, 1, algorithms, test_type=TestType.FULL_TRAINING)

print(comparer)
print(comparer.last_confusion)

print("Age: 60; Cholesterol: 199; Duration: 2")
print(lr.evaluate({"age": 60, "choleste": 199, "dur": 2}))

print("With sex.")

ds = db.create_heart_dataset()
algorithms = []
algorithms.append(LogisticalRegression(solver='lbfgs'))
algorithms.append(KNN())
algorithms.append(KNN(weighted=True))
algorithms.append(KNearest(2))

comparer = Comparer(ds, 1, algorithms, test_type=TestType.FULL_TRAINING)

print(comparer)
print(comparer.last_confusion)
