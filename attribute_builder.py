from collections import Counter


blackListedWords = ["de", "la", "en", "el", "que", "y", "los", "un", "del", "al", "fue", "es", "san"
                    "con", "para", "se", "una", "su", "a", "más", "por", "las", "no", "le", "con", "porque", "casi", "desde"
                    , "|", "tras", "sobre", "sus", "qué", "son", "entre", "pero", "hay", "ya", "como", "desde", "hasta", "me", "lo", "san", "tu", "tus", "cómo"]

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

def CountFrequencies(str, amount):
    cleanStr = str.lower().split(" ")
    return Counter(x for x in cleanStr if x not in blackListedWords).most_common(amount)

def GetExampleAttrs():
    return  {'categoria': ['count'], 'titular' : {"dolar": lambda x: MakeList(x, "dolar"),  "BRCA": lambda x: MakeList(x, "BRCA")}}

def buildNMostCommonWordsByCategory(dataframe, groupByKey, amount):
    # Uncomment to find out the most frequent words
    mostCommon = dataframe.groupby(groupByKey).agg({lambda x: CountFrequencies(ConcatTitles(x), amount)})
    mostCommon = mostCommon.agg({lambda x: [row[0] for row in x]})
    words = []
    for asd in mostCommon.itertuples():
        words.append(asd[1])
    return [item for sublist in words for item in sublist]