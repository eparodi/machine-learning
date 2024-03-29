import os, sys;

import time
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from common.algorithms.kohonen import Kohonen

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

alg = Kohonen(rounds=10, n=20, distance_weight=0.5, iteration_weight=0.5)
alg.train(dataset)
nodes_df = alg.get_nodes_df()
print(nodes_df)

nodes_df['color'] = nodes_df[Kohonen.NODE_CLASS].replace({
    "Pagni": "blue",
    "VanderKooy": "red",
    "Calderaro": "green",
    "Verbitsky": "yellow",
    "Fonteveccia": "brown",
})

ax = plt.gca()
nodes_df.plot(kind='scatter',x=Kohonen.NODE_X, y=Kohonen.NODE_Y,s=30, marker="*", color=nodes_df["color"], ax=ax)
plt.show()
# print(dataset)

