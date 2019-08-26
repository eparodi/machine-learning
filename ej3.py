from naive_bayes import NaiveBayes

import attribute_builder as attrs
import pandas as pd
import xlrd as xl
import re

wordCountAttrs = ["Google", "WhatsApp", "nuevo","River", "Boca", "final", "San", "contra", "Gobierno", "dólar", "millones", "baja",
                  "Pérez", "Sol", "foto", "contra", "Trump", "príncipe", "mujer", "años", "cáncer", "VIH"]
attributes = ({'categoria': ['count'] })
titular = {}
for itup in wordCountAttrs:
    titular[itup]=attrs.buildWordFrequencyAttr(itup)
# Uncomment to find out the most frequent words
# titular["MostFrequentWords"]= lambda x: attrs.CountFrequencies(attrs.ConcatTitles(x))
attributes['titular'] = titular

def split(trainingPercentage, dataset):
    middleIndex = int(len(allNews) * trainingPercentage)
    return (allNews[:middleIndex], allNews[middleIndex:])

def maxDictItem(map):
    maxCat = ""
    maxCatConfidence = 0
    for category, confidence in map.items():
        if(maxCatConfidence < confidence):
            maxCat = category
            maxCatConfidence = confidence
    return maxCat


allNews = pd.read_csv('news.tsv', sep='\t', header=0)
allNews = allNews.drop('fuente', axis=1).drop('fecha', axis=1)
training, test = split(0.5, allNews)
print(len(allNews))
print(len(training))
print(len(test))

for title in wordCountAttrs:
    training[title] = training.titular.str.count(title, flags=re.IGNORECASE)

news = training.drop('titular', axis=1)
bayes = NaiveBayes.from_data_frame(news, 'categoria')

truePositive = 0
trueNegative = 0
total = 0

for asd in test.itertuples():
    inp = [1 if word.upper() in asd.titular.upper() else 0 for word in wordCountAttrs]
    result = bayes.get_probabilities(inp)
    total+=1
    if(maxDictItem(result) == asd.categoria):
        truePositive+=1
    else:
        trueNegative+=1

print(truePositive/total)