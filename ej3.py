from naive_bayes import NaiveBayes

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

training, test = exp.random_with_replacement_split(0.8, allNews)
print('Training size:'+ str(len(training)))
print('Test size:'+ str(len(test)))
print('Total size:'+ str(len(allNews)))

mostCommonAmount = 7
autoDetectedAttrs = attrs.buildNMostCommonWordsByCategory(training, 'categoria', mostCommonAmount)
print("Detecting the " + str(mostCommonAmount) + " most common words for each category")
usedAttrs = autoDetectedAttrs

print("Using these attributes: " + str(usedAttrs))
for attribute in usedAttrs:
    training[attribute] = training.titular.str.count(attribute, flags=re.IGNORECASE)

news = training.drop('titular', axis=1)
bayes = NaiveBayes.from_data_frame(news, 'categoria')

truePositive = 0
trueNegative = 0
total = 0

actualCategory = []
guessedCategory = []

for asd in test.itertuples():
    inp = [1 if word.lower() in asd.titular.lower() else 0 for word in usedAttrs]
    # print(asd)
    # print(inp)
    result = bayes.get_probabilities(inp)
    total+=1
    guessed = exp.maxDictItem(result)
    actualCategory.append(asd.categoria)
    guessedCategory.append(guessedCategory)

    if(guessed == asd.categoria):
        # print("Correctly guessed!")
        truePositive+=1
    else:
        # print("Mistaken")
        trueNegative+=1


y_actu = pd.Series(actualCategory, name='Actual Category')
y_pred = pd.Series(guessedCategory, name='Predicted Category')
df_confusion = pd.crosstab(y_actu, y_pred)

print(truePositive/total)
print(df_confusion)