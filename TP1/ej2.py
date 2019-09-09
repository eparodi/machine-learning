import pandas as pd

from common.naive_bayes import NaiveBayes

britons = pd.ExcelFile('datasets/britons.xlsx').parse()
inp = (1, 0, 1, 1, 0)
bayes = NaiveBayes.from_data_frame(britons, "Nacionalidad")
print(bayes.get_probabilities(inp))

