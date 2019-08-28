from naive_bayes import NaiveBayes

import math
import attribute_builder as attrs
import pandas as pd
import re
import experiment_builder as exp

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

manualAttributes = ["Google", "WhatsApp", "nuevo", "River", "Boca", "final", "San", "contra", "Gobierno", "dólar", "millones", "baja",
                  "Pérez", "Sol", "foto", "contra", "Trump", "príncipe", "mujer", "años", "cáncer", "VIH", "sarampión"]

allNews = pd.read_csv('news.tsv', sep='\t', header=0)
allNews = allNews.drop('fuente', axis=1).drop('fecha', axis=1)
allNews = allNews[allNews.categoria!='Noticias destacadas']
allNews = allNews[allNews.categoria!='Nacional']
allNews = allNews[allNews.categoria!='Internacional']
allNews = allNews[allNews.categoria!='Destacadas']
allNews = allNews[allNews.categoria!='Entretenimiento']
allNews = allNews.dropna()

allCategories = []
for category in allNews.groupby('categoria').nunique().itertuples():
    allCategories.append(category[0])

training, test = exp.random_with_replacement_split(0.3, allNews)
# training, test = (allNews.copy(), allNews.copy())
print('Training size:'+ str(len(training)))
print('Test size:'+ str(len(test)))
print('Total size:'+ str(len(allNews)))

mostCommonAmount = 25
autoDetectedAttrs = attrs.buildNMostCommonWordsByCategory(training, 'categoria', mostCommonAmount)
print("Detecting the " + str(mostCommonAmount) + " most common words for each category")
usedAttrs = autoDetectedAttrs

print("Using these categories: " + str(allCategories))
print("Using these attributes: " + str(usedAttrs))
for attribute in usedAttrs:
    training[attribute] = training.titular.str.count(attribute, flags=re.IGNORECASE)

news = training.drop('titular', axis=1)
bayes = NaiveBayes.from_data_frame(news, 'categoria')

hit = 0
miss = 0
total = 0

confusion = pd.DataFrame(data=0, index=allCategories, columns=allCategories)
metrics = pd.DataFrame(data=0, index=allCategories, columns=["Accuracy", "Precision", "Recall", "F1-score", "TruePositives", "TrueNegatives", "FalsePositives", "FalseNegatives", "Total"])

def addOneToDataframe(dataframe, category, metric):
    dataframe.loc[category, metric] = int(metrics.loc[category, metric]) + 1


for row in test.itertuples():
    inp = [1 if word.lower() in row.titular.lower() else 0 for word in usedAttrs]
    result = bayes.get_probabilities(inp)
    total+=1
    guessed = exp.maxDictItem(result)
    categoria = row.categoria
    # print(confusion)
    try:
        oldValue = confusion.loc[categoria, guessed]
        confusion.loc[categoria, guessed] = int(oldValue) + 1

    except KeyError:
        print("ERROR!")
        print(row)

    addOneToDataframe(metrics, categoria, "Total")
    if(guessed == row.categoria):
        hit += 1
    else:
        miss += 1

confusion.loc['Column_Total']= confusion.sum(numeric_only=True, axis=0)
confusion.loc[:,'Row_Total'] = confusion.sum(numeric_only=True, axis=1)


for category in allCategories:
    metrics.loc[category,"FalsePositives"] = confusion.loc["Column_Total", category] - confusion.loc[category, category]
    metrics.loc[category,"TruePositives"] = confusion.loc[category, category]
    metrics.loc[category,"FalseNegatives"] = confusion.loc[category, "Row_Total"] - confusion.loc[category, category]
    metrics.loc[category,"TrueNegatives"] = confusion.loc["Column_Total", "Row_Total"] - confusion.loc[category, "Row_Total"]

metrics["Accuracy"] = (metrics["TruePositives"] + metrics["TrueNegatives"])/confusion.loc["Column_Total", "Row_Total"]
metrics["Precision"] = metrics["TruePositives"] / (metrics["TruePositives"] + metrics["FalsePositives"])
metrics["Recall"] = metrics["TruePositives"] / (metrics["TruePositives"] + metrics["FalseNegatives"])
metrics["F1-score"] = (2*metrics["Precision"]*metrics["Recall"]) / (metrics["Precision"] + metrics["Recall"])

print("\nConfusion Matrix:")
print(confusion)
print("\nMetrics by Category:")
print(metrics)
print("\nCorrectly categorized: " + str(hit/total))