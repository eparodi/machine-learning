import math as m
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
