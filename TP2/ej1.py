import pandas as pd
import math as m

from common.information import inf_gain
from common.tree import Tree

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

sports = pd.read_csv('../datasets/juegaTenis.csv', sep=',', header=0)
class_col = 'Juega'
blackListedCols = ['Dia']

allCategories =list(sports)
allCategories.remove(class_col)
for blackListedCol in blackListedCols:
    allCategories.remove(blackListedCol)

def evaluateNextNode(data_frame: pd.DataFrame, categories):
    discriminator = data_frame.copy().groupby(class_col).agg(count=(class_col, 'count'))
    # print("discriminador:\n" + str(discriminator))
    head = discriminator.head().index.values
    if(len(head) == 1):
        return Tree(head[0])
    else:
        categoryData = pd.DataFrame(data=0, index=categories, columns=["InfGain"])
        for category in categories:
            categoryData.loc[category, "InfGain"] = inf_gain(data_frame, category, class_col)
        # print("CategoryData\n" + str(categoryData))
        mostLikelyCategory = categoryData["InfGain"].idxmax()
        possibleValues = data_frame.groupby(mostLikelyCategory).agg(count=(class_col, 'count')).head().index.values
        categories.remove(mostLikelyCategory)
        leafs = []
        for possibleValue in possibleValues:
            cut_data_frame = data_frame[data_frame[mostLikelyCategory] == possibleValue]
            leafs.append(Tree(possibleValue, evaluateNextNode(cut_data_frame, categories)))
        return Tree(mostLikelyCategory, leafs)

print(sports)
print(evaluateNextNode(sports, allCategories))

