from collections import namedtuple
from enum import Enum, auto

import pandas as pd

from common.dataset_manager import Dataset
from common.tree import Tree
from common.information import inf_gain

class NodeType(Enum):
    ATTR = auto()
    VALUE = auto()
    CLAZZ = auto()

class Node:
    def __init__(self, value, type):
        self.value = value
        self.type = type

    def __str__(self):
        if self.type == NodeType.ATTR:
            return "A:" + str(self.value)
        elif self.type == NodeType.VALUE:
            return "V:" + str(self.value)
        else:
            return "C:" + str(self.value)

class DecisionTree:

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
            return Tree(Node(head[0], NodeType.CLAZZ))
        elif len(categories) == 0:
            # This means there are different possible classes but there are no more categories to expand
            return Tree(Node(str(class_col_freq.idxmax()), NodeType.CLAZZ))
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
                return Tree(Node("kill", NodeType.CLAZZ))
            possibleValues = data_frame.groupby(mostLikelyCategory).agg(count=(class_col, 'count')).index.values
            categories.remove(mostLikelyCategory)
            leafs = []
            for possibleValue in possibleValues:
                cut_data_frame = data_frame[data_frame[mostLikelyCategory] == possibleValue]
                leafs.append(Tree(Node(possibleValue, NodeType.VALUE), self.evaluateNextNode(cut_data_frame, categories, class_col)))
            return Tree(Node(mostLikelyCategory, NodeType.ATTR), leafs)

    def get_tree(self):
        return self.tree

    def evaluate(self, row_dict):
        node = self.tree
        prevNode = 0
        print(row_dict)
        # Sometimes it hangs because there is some value that was not there during training and the algoritm does not take it into account
        while node.value.type != NodeType.CLAZZ and len(node.leafs) != 0:
            if node == prevNode:
                raise AssertionError("LOOPING for " + node)
            value = node.value.value
            type = node.value.type
            if type == NodeType.ATTR:
                attrValue = row_dict[value]
                for child in node.leafs:
                    if child.value.value == attrValue:
                        prevNode = node
                        node = child
                    if node == prevNode:
                        raise AssertionError("LOOPING for " + node)
            elif type == NodeType.VALUE:
                if len(node.leafs) == 0:
                    raise AssertionError("No continued attr after Value for " + value)
                else:
                    prevNode = node
                    node = node.leafs[0]
            elif type == NodeType.CLAZZ:
                return node.value.value
            else:
                raise AssertionError("Not a valid NodeType!: " + str(type))

        return node.value.value
