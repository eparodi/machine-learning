import os, sys; sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import common.dataset.dataset_builder as db
from common.algorithms.simple_perceptron import SimplePerceptron
from common.experiment.experiment_comparer import Comparer, TestType
from numpy import linspace

ds = db.create_linearly_separable_dataset(height=1, width=1, n=50)

algorithms = [SimplePerceptron(epochs=epoch, learning_rate=lrate)
              for epoch in [0, 1, 5, 10]
              for lrate in [0.1, 0.3, 0.6, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]]

comparer = Comparer(ds, 0.8, algorithms, rounds=10, test_type=TestType.DISJOINT)

print(comparer)

print(algorithms[-1].weight)
print(algorithms[-1].attrOrder)
weights = algorithms[-1].weight
ax = plt.gca()
ds.getRows().plot(kind='scatter',x='x',y="y", color=ds.getRows()["color"], ax=ax)

x = linspace(-0.5, 0.5, 400)
y = linspace(
    (weights[0] + weights[1] * 0.5) / weights[2],
    (weights[0] - weights[1] * 0.5) / weights[2],
    400)
plt.plot(x, y)

plt.ylim((-0.5, 0.5))
plt.xlim((-0.5, 0.5))

plt.show()