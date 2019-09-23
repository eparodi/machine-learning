import pandas as pd

def get_class_probs(data_frame:pd.DataFrame, class_col):
    class_col_counts = data_frame[class_col].value_counts()
    return class_col_counts / class_col_counts.sum()