import os, sys;

import time
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

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

df = dataset.getRows()
df['journalist'] = df['journalist'].replace({
    "Pagni": "blue",
    "VanderKooy": "red",
    "Calderaro": "green",
    "Verbitsky": "yellow",
    "Fonteveccia": "brown",
})



print(dataset)

