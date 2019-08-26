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

def GetExampleAttrs():
    return  {'categoria': ['count'], 'titular' : {"dolar": lambda x: MakeList(x, "dolar"),  "BRCA": lambda x: MakeList(x, "BRCA")}}
