import pandas as pd

from common.dataset_manager import Dataset
from common.naive_bayes import NaiveBayes


ds = Dataset.create_britons_dataset()
britons = ds.getRows()
inp = (1, 0, 1, 1, 0)
bayes = NaiveBayes.from_data_frame(britons, "Nacionalidad")
print(bayes.get_probabilities(inp))

