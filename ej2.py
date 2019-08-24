from naive_bayes import NaiveBayes

import pandas as pd

britons = pd.ExcelFile('britons.xlsx').parse()
inp = (1, 0, 1, 1, 0)
bayes = NaiveBayes.from_data_frame(britons, "Nacionalidad")
print(bayes.get_probabilities(inp))
