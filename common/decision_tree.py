import pandas as pd

from common.dataset_manager import Dataset
from common.tree import Tree
from common.information import inf_gain


class DecisionTree():

    def __init__(self,  data_frame: pd.DataFrame, categories, class_col, min_samples=5):
        self.data_frame = data_frame.copy()
        self.categories = categories.copy()
        self.class_col = class_col
        self.min_samples = min_samples
        self.tree = self.evaluateNextNode(self.data_frame, self.categories, self.class_col)

    @classmethod
    def from_dataset(self, dataset: Dataset, min_samples=5):
        return DecisionTree(dataset.getRows(), dataset.getAttributes(), dataset.getClassAttr(), min_samples)

    def evaluateNextNode(self, data_frame: pd.DataFrame, categories, class_col):
        class_col_counts = data_frame.copy()[class_col].value_counts()
        class_col_freq = class_col_counts / class_col_counts.sum()
        head = class_col_freq.head().index.values
        if len(head) == 1:
            return Tree(head[0])
        elif len(categories) == 0:
            # This means there are different possible classes but there are no more categories to expand
            return Tree(str(class_col_freq.idxmax()))
        else:
            categoryData = pd.DataFrame(data=0, index=categories, columns=["InfGain"])
            categories = categories.copy()
            for category in categories:
                categoryData.loc[category, "InfGain"] = inf_gain(data_frame, category, class_col)
            if len(categoryData["InfGain"].values) == 0:
                print("error")
            mostLikelyCategory = categoryData["InfGain"].idxmax()
            if (categoryData["InfGain"][mostLikelyCategory] <= 0.0001):
                categories.remove(mostLikelyCategory)
                return Tree("kill")
            possibleValues = data_frame.groupby(mostLikelyCategory).agg(count=(class_col, 'count')).index.values
            categories.remove(mostLikelyCategory)
            leafs = []
            for possibleValue in possibleValues:
                cut_data_frame = data_frame[data_frame[mostLikelyCategory] == possibleValue]
                leafs.append(Tree(possibleValue, self.evaluateNextNode(cut_data_frame, categories, class_col)))

            return Tree(mostLikelyCategory, leafs)

    def get_tree(self):
        return self.tree