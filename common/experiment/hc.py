import os, sys; sys.path.append(os.getcwd())
import common.dataset.dataset_builder as db

import time
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from common.algorithms.hierarchical_clustering import HierarchicalClustering

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


dataset = db.create_clustered_dataset(
    clusters=5, points_per_cluster=20, cluster_size=0.1)

alg = HierarchicalClustering(method=HierarchicalClustering.CENTROID, n=5)
alg.train(dataset)
nodes = alg.get_groups(5)

colors = {1: "green", 2: "red", 0: "blue", 3: "brown", 4: "yellow"}
flat_nodes = []
for i in range(0, 5):
    for d in nodes[i]["data"]:
        d["color"] = colors[i]
        flat_nodes.append(d)

nodes_df = pd.DataFrame(flat_nodes, columns=[
                        "x", "y", "color"])
df = dataset.getRows()
ax = plt.gca()
df.plot(kind='scatter', x='x', y="y", s=5,
        color=dataset.getRows()["color"], ax=ax)
plt.show()
ax = plt.gca()
nodes_df.plot(kind='scatter', x='x', y="y", s=30, marker="*",
              color=nodes_df["color"], ax=ax)
plt.show()