import os, sys;

import time
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from common.algorithms.kohonen import Kohonen
from common.experiment.experiment_comparer import Comparer, TestType

sys.path.append(os.getcwd())

from common.dataset.journalist_dataset import create_journalist_dataset

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
def get_journalists(data):
    journalists = [d["journalist"] for d in data]
    j_dict = dict.fromkeys(journalists)
    for j in j_dict.keys():
        j_dict[j] = journalists.count(j)
    print(j_dict)

dataset = create_journalist_dataset()

algorithms = [Kohonen(rounds=rounds, n=n, distance_weight=0.5, iteration_weight=0.5)
              for rounds in [2]
              for n in [5, 10, 20]]

comparer = Comparer(dataset, 0.5, algorithms, rounds=10, test_type=TestType.DISJOINT)

print(comparer)

