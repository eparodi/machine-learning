from random import sample

from common.algorithms.algorithm import Algorithm
from common.algorithms.decision_tree import DecisionTree
from common.dataset.dataset import Dataset
from common.utils.information import InfGainFunction


class RandomForest(Algorithm):
    def __init__(self, n_trees=5, data_used=0.6, attrs_used=0.8, inf_gain_function=InfGainFunction.SHANNON):
        self.is_trained = False
        self.attrs_used = attrs_used
        self.data_used = data_used
        self.n_trees = n_trees
        self.inf_gain_function = inf_gain_function

    def is_trained(self):
        return self.is_trained

    def train(self, dataset: Dataset):
        self.data_frame = dataset.getRows().copy()
        self.categories = dataset.getAttributes().copy()
        self.class_col = dataset.getClassAttr()
        self.trees = RandomForest.build_trees()
        self.is_trained = True

    def get_tags(self):
        return {"Algorithm": "RandomForest",
                "N_Trees": self.n_trees,
                "data_used": self.data_used,
                "attrs_used": self.attrs_used
                }

    def evaluate(self, values_dict: dict):
        pass

    def build_trees(self):
        trees = []
        for i in range(0, self.n_trees):
            tree_data = self.data_frame.sample(frac=self.data_used, )
            tree_categories = sample(categories, len(categories)*categories_used)
            trees.append(DecisionTree(tree_data, tree_categories, class_col))
        return trees
