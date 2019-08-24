import pandas as pd
import xlrd as xl
import re
from collections import Counter

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

britons = pd.read_csv('news.tsv', sep='\t', header=0)
exampleAttrs = {'categoria': ['count'], 'titular' : {"dolar": lambda x: MakeList(x, "dolar"),  "BRCA": lambda x: MakeList(x, "BRCA")}}
print(britons)
print(attributes)
probs = britons.groupby('categoria').agg(attributes)
print(probs)

# probs_britons = britons.groupby('Nacionalidad').agg({'Nacionalidad': ['count']})
# probs_britons = probs_britons.div(probs_britons.sum())
#
# probs = probs.to_dict()
# probs_britons = probs_britons.to_dict()
# probs_britons = probs_britons[list(probs_britons.keys())[0]]
#
# inp = (1, 0, 1, 1, 0)
# keys = list(probs.keys())
#
# for _cls in probs_britons.keys():
#     prob = 1
#     for x in range(len(keys)):
#         i_value = inp[x]
#         key = keys[x]
#         if i_value:
#             prob *= probs[key][_cls]
#         else:
#             prob *= (1 - probs[key][_cls])
#     prob *= probs_britons[_cls]
#     print("{_cls}: {prob}".format(_cls=_cls, prob=prob))

