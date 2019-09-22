from enum import Enum, auto

import pandas as pd
import common.utils.strings as s

from common.algorithms.algorithm import Algorithm
from common.dataset.dataset import Dataset
from common.utils.information import InfGainFunction
from common.utils.tree import Tree


class NodeType(Enum):
    ATTR = auto()
    VALUE = auto()
    CLAZZ = auto()


class Node:
    def __init__(self, value, type, probs: dict):
        self.value = value
        self.type = type
        self.probs = probs

    def __str__(self):
        if self.type == NodeType.ATTR:
            return "A:" + str(self.value) + " \t\tProbs:" + s.one_line_dict(self.probs)
        elif self.type == NodeType.VALUE:
            return "V:" + str(self.value) + " \t\tProbs:" + s.one_line_dict(self.probs)
        else:
            return "C:" + str(self.value) + " \t\tProbs:" + s.one_line_dict(self.probs)

    @classmethod
    def fromAttr(cls, value, probs: dict):
        return Node(value, NodeType.ATTR, probs)
    @classmethod
    def fromValue(cls, value, probs: dict):
        return Node(value, NodeType.VALUE, probs)
    @classmethod
    def fromClazz(cls, value, probs: dict):
        return Node(value, NodeType.CLAZZ, probs)


class DecisionTree(Algorithm):

    def __init__(self, min_samples=5, inf_gain_function=InfGainFunction.SHANNON, used_percentage=1):
        super().__init__()
        self.used_percentage = used_percentage
        self.min_samples = min_samples
        self.inf_gain_function = inf_gain_function

    def get_tags(self):
        return {s.algorithm: "DecisionTree",
                "Used": self.used_percentage,
                "InfGain": self.inf_gain_function.name}

    def train(self, dataset: Dataset):
        used_ds = dataset
        if self.used_percentage != 1:
            used_ds = dataset.build_random_sample_dataset(self.used_percentage)

        self.data_frame = used_ds.getRows().copy()
        self.categories = used_ds.getAttributes().copy()
        self.class_col = used_ds.getClassAttr()
        self.tree = self.evaluateNextNode(self.data_frame, self.categories, self.class_col)
        self.is_trained = True
        print(self.tree)

    def evaluateNextNode(self, data_frame: pd.DataFrame, categories, class_col):
        class_col_counts = data_frame.copy()[class_col].value_counts()
        class_col_freq = class_col_counts / class_col_counts.sum()
        most_likely_class = class_col_freq.idxmax()
        head = class_col_freq.head().index.values
        if len(head) == 1 or len(categories) == 0:
            # This means there are different possible classes but there are no more categories to expand
            return Tree(Node.fromClazz(str(most_likely_class), class_col_freq))
        else:
            categoryData = pd.DataFrame(data=0, index=categories, columns=["InfGain"])
            categories = categories.copy()
            for category in categories:
                categoryData.loc[category, "InfGain"] = self.inf_gain_function.value.func(data_frame, category, class_col)
            if len(categoryData["InfGain"].values) == 0:
                print("error")
            most_informational_category = categoryData["InfGain"].idxmax()
            if (categoryData["InfGain"][most_informational_category] <= 0.0001):
                categories.remove(most_informational_category)
                return Tree(Node.fromClazz(str(most_likely_class), class_col_freq))
            possibleValues = data_frame.groupby(most_informational_category).agg(count=(class_col, 'count')).index.values
            categories.remove(most_informational_category)
            leafs = []
            for possibleValue in possibleValues:
                cut_data_frame = data_frame[data_frame[most_informational_category] == possibleValue]
                leafs.append(Tree(Node.fromValue(possibleValue, class_col_freq), self.evaluateNextNode(cut_data_frame, categories, class_col)))
            return Tree(Node.fromAttr(most_informational_category, class_col_freq), leafs)

    def get_tree(self):
        return self.tree

    def evaluate(self, values_dict):
        node = self.tree
        prevNode = 0
        # Sometimes it hangs because there is some value that was not there during training and the algoritm does not take it into account
        while node.value.type != NodeType.CLAZZ and len(node.leafs) != 0:
            if node == prevNode:
                raise AssertionError("LOOPING for " + node)
            value = node.value.value
            type = node.value.type
            if type == NodeType.ATTR:
                attrValue = values_dict[value]
                attr_value_exists_in_tree = False
                for child in node.leafs:
                    if child.value.value == attrValue:
                        prevNode = node
                        node = child
                        attr_value_exists_in_tree = True
                if not attr_value_exists_in_tree:
                    return node.value.probs.idxmax()
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



