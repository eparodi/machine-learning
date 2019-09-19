import re

import pandas as pd

from common import attribute_builder as attrs, experiment_builder as exp
from common.dataset_manager import Dataset
from common.experiment_evaluator import Test, Evaluator
from common.naive_bayes import NaiveBayes

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

manualAttributes = ["Google", "WhatsApp", "nuevo", "River", "Boca", "final", "San", "contra", "Gobierno", "dólar", "millones", "baja",
                  "Pérez", "Sol", "foto", "contra", "Trump", "príncipe", "mujer", "años", "cáncer", "VIH", "sarampión"]

ds = Dataset.create_news_dataset()
class_col, allCategories, allNews = ds.getClassAttr(), ds.getClassAttrValues(), ds.getRows()

allNews = allNews[allNews.categoria!='Noticias destacadas']
allNews = allNews[allNews.categoria!='Nacional']
allNews = allNews[allNews.categoria!='Internacional']
allNews = allNews[allNews.categoria!='Destacadas']
allNews = allNews[allNews.categoria!='Entretenimiento']
allNews = allNews.dropna()

training, test = exp.random_with_replacement_split(0.8, allNews)
# training, test = (allNews.copy(), allNews.copy())
print('Training size:'+ str(len(training)))
print('Test size:'+ str(len(test)))
print('Total size:'+ str(len(allNews)))

mostCommonAmount = 20
autoDetectedAttrs = attrs.buildNMostCommonWordsByCategory(training, class_col, mostCommonAmount)
print("Detecting the " + str(mostCommonAmount) + " most common words for each category")
usedAttrs = autoDetectedAttrs

print("Using these categories: " + str(allCategories))
print("Using these attributes: " + str(usedAttrs))
for attribute in usedAttrs:
    training[attribute] = training.titular.str.count(attribute, flags=re.IGNORECASE)

news = training.drop('titular', axis=1)
bayes = NaiveBayes.from_data_frame(news, 'categoria')

testResult = []
for row in test.itertuples():
    inp = [1 if word.lower() in row.titular.lower() else 0 for word in usedAttrs]
    result = bayes.get_probabilities(inp)
    guessed = exp.maxDictItem(result)
    actual = row.categoria
    testResult.append(Test(actual, guessed))

print(Evaluator(testResult, allCategories))