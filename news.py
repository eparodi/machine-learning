from naive_bayes import NaiveBayes
from collections import Counter

import pandas as pd
import xlrd as xl
import re

blackListedWords = ["de", "la", "en", "el", "que", "y", "los", "un", "del", "al", "fue", "es", "lo"
                    "con", "para", "se", "una", "su", "a", "más", "por", "las", "no", "le", "con",
                    "lo", "|", "tras", "sobre", "sus", "qué"]

def buildWordFrequencyAttr(str):
    return lambda x: MakeList(x, str)

def Count(str, term):
    return str.upper().count(term.upper())

def MakeList(x, term):
    T = tuple(x)
    return sum(tuple(Count(itup, term) for itup in T))

def ConcatTitles(x):
    T = tuple(x)
    str = ''
    for itup in T:
        str += itup
    return str

def CountFrequencies(str):
    cleanStr = str.split(" ")
    return Counter(x for x in cleanStr if x not in blackListedWords).most_common(5)

wordCountAttrs = ["Google", "WhatsApp", "nuevo","River", "Boca", "final", "San", "contra", "Gobierno", "dólar", "millones", "baja",
                  "Pérez", "Sol", "foto", "contra", "Trump", "príncipe", "mujer", "años", "cáncer", "VIH"]
attributes = ({'categoria': ['count'] })
titular = {}
for itup in wordCountAttrs:
    titular[itup]=buildWordFrequencyAttr(itup)
# Uncomment to find out the most frequent words
# titular["MostFrequentWords"]= lambda x: CountFrequencies(ConcatTitles(x))

attributes['titular'] = titular

news = pd.read_csv('news.tsv', sep='\t', header=0)
news = news.drop('fuente', axis=1).drop('fecha', axis=1)
exampleAttrs = {'categoria': ['count'], 'titular' : {"dolar": lambda x: MakeList(x, "dolar"),  "BRCA": lambda x: MakeList(x, "BRCA")}}
for title in wordCountAttrs:
    news[title] = news.titular.str.count(title, flags=re.IGNORECASE)

news = news.drop('titular', axis=1)
bayes = NaiveBayes.from_data_frame(news, 'categoria')

title = 'Google sube el dolar contra Trump'

inp = [1 if word.upper() in title.upper() else 0 for word in wordCountAttrs]
print(bayes.get_probabilities(inp))