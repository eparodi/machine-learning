import pandas as pd

from common.dataset.dataset import Dataset
from common.experiment import experiment_builder as exp
from common.experiment.experiment_analizer import TestsAnalizer, Test
from common.naive_bayes import NaiveBayes

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

ds = Dataset.create_britons_dataset()
class_col, class_values, attrs, rows = ds.getClassAttr(), ds.getClassAttrValues(), ds.getAttributes(), ds.getRows()
training, test = exp.random_with_replacement_split(0.5, rows)
print(training)
print(test)

bayes = NaiveBayes.from_data_frame(training, class_col)

tests = []
for row in test.itertuples(index=False):
    probs = bayes.get_probabilities(row)
    guessed = max(probs, key=probs.get)
    actual = row._asdict()[class_col]
    tests.append(Test(actual, guessed))

print(TestsAnalizer(tests, class_values))


