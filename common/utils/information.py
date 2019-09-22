import math as m
from collections import namedtuple
from enum import Enum
from typing import NamedTuple

import pandas as pd

def entropyPart(x):
    return -m.log2(x)*x

def entropy(data_frame: pd.DataFrame, class_col: str):
    data_frame = data_frame.copy()
    data_frame = data_frame.groupby(class_col).agg(count=(class_col, 'count'))
    data_frame["Total"] = data_frame["count"].sum()
    data_frame["Freq"] = data_frame["count"] / data_frame["Total"]
    data_frame['Entropy'] = data_frame["Freq"].apply(entropyPart)
    return data_frame["Entropy"].sum()

def inf_gain(data_frame: pd.DataFrame, category: str, class_col: str):
    categories = data_frame.copy().groupby(category).agg(Size=(class_col, 'count'))
    parentEntropy = entropy(data_frame, class_col)
    # categories["ParentEntropy"] = entropy(data_frame, class_col)
    categories["ParentSize"] = len(data_frame)
    categories["Weight"] = categories["Size"] / categories["ParentSize"]
    for row in categories.itertuples():
        categories.loc[row[0],"Entropy"] = entropy(data_frame[data_frame[category]==row[0]], class_col)
    categories["WeighedEntropy"] = categories["Weight"] * categories["Entropy"]
    # print(categories)
    return parentEntropy - categories["WeighedEntropy"].sum()

# No entiendo como hacer Gini :(
# No usar que esta roto
def gini(data_frame: pd.DataFrame, category: str, class_col: str):
    class_col_counts = data_frame.copy()[class_col].value_counts()
    class_col_freq = class_col_counts / class_col_counts.sum()
    most_likely_class = class_col_freq.idxmax()

InfGainFunc = namedtuple('InfGainFunc', 'name func')


class InfGainFunction(Enum):
    SHANNON = InfGainFunc("SHANNON", inf_gain)
    GINI = InfGainFunc("GINI", inf_gain)