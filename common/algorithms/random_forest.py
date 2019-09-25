from random import sample

from common.algorithms.algorithm import Algorithm
from common.algorithms.decision_tree import DecisionTree
from common.dataset.dataset import Dataset
from common.utils.information import InfGainFunction


class RandomForest(Algorithm):
    def __init__(self, n_trees=5, data_used=0.5, max_nodes=4,attrs_used=0.5, inf_gain_function=InfGainFunction.SHANNON):
        super().__init__()
        self.attrs_used = attrs_used
        self.data_used = data_used
        self.n_trees = n_trees
        self.inf_gain_function = inf_gain_function
        self.max_nodes = max_nodes

    def train(self, dataset: Dataset):
        self.dataset = dataset
        self.categories = dataset.getAttributes()
        self.trees = self.build_trees()
        for tree in self.trees:
            print(str(tree))

    def get_tags(self):
        return {"Algorithm": "RandomForest",
                "N_Trees": self.n_trees,
                "data_used": self.data_used,
                "attrs_used": self.attrs_used
                }

    def evaluate(self, values_dict: dict):
        votes = {}
        for clazz_value in self.dataset.getClassAttrValues():
            votes[clazz_value] = 0
        for tree in self.trees:
            result = tree.evaluate(values_dict=values_dict)
            votes[result] += 1
        return max(votes, key=votes.get)

    def build_trees(self):
        trees = []
        for i in range(0, self.n_trees):
            tree_dataset = self.dataset.build_random_sample_dataset(self.data_used)
            tree_categories = sample(self.dataset.getAttributes(), int(len(self.categories)*self.attrs_used))
            tree = DecisionTree(max_nodes=self.max_nodes, categories=tree_categories, inf_gain_function=self.inf_gain_function)
            tree.train(tree_dataset)
            trees.append(tree)
        return trees

