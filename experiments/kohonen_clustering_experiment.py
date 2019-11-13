import os, sys;

import time
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from common.algorithms.kohonen import Kohonen

sys.path.append(os.getcwd())


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

import common.dataset.dataset_builder as db

dataset = db.create_clustered_dataset(clusters=4, points_per_cluster=25, cluster_size=0.1)

alg = Kohonen(rounds=50, n=15, distance_weight=0.5, iteration_weight=0.5)
alg.train(dataset)

nodes_df = alg.get_nodes_df()
print(nodes_df)

df = dataset.getRows()
ax = plt.gca()
df.plot(kind='scatter',x='x',y="y",s=5, color=dataset.getRows()["color"], ax=ax)
nodes_df.plot(kind='scatter',x='x',y="y",s=30, marker="*", color=nodes_df["NODE_CLASS"], ax=ax)
plt.show()

