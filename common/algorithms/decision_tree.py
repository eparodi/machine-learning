from enum import Enum, auto

import pandas as pd
import common.utils.strings as s

from common.algorithms.algorithm import Algorithm
from common.dataset.dataset import Dataset
from common.utils.information import InfGainFunction
from common.utils.probs import get_class_probs
from common.utils.tree import Tree


class NodeType(Enum):
    ATTR = auto()
    VALUE = auto()
    CLAZZ = auto()


class Node:
    def __init__(self, value, type, probs: dict, size):
        self.value = value
        self.type = type
        self.probs = probs
        self.size = size

    def __str__(self):
        if self.type == NodeType.ATTR:
            return "A:" + self.node_to_string()
        elif self.type == NodeType.VALUE:
            return "V:" + self.node_to_string()
        else:
            return "C:" + self.node_to_string()

    def node_to_string(self):
        return str(self.value) + "\t\t\tSize:" + str(self.size) + " Probs:" + str(s.one_line_dict(self.probs))

    @classmethod
    def fromAttr(cls, value, probs: dict, size):
        return Node(value, NodeType.ATTR, probs, size)
    @classmethod
    def fromValue(cls, value, probs: dict, size):
        return Node(value, NodeType.VALUE, probs, size)
    @classmethod
    def fromClazz(cls, value, probs: dict, size):
        return Node(value, NodeType.CLAZZ, probs, size)


class DecisionTree(Algorithm):

    def __init__(self, min_samples=None, max_nodes=None, max_depth=None, inf_gain_function=InfGainFunction.SHANNON, used_percentage=1):
        super().__init__()
        self.used_percentage = used_percentage
        self.min_samples = min_samples
        self.inf_gain_function = inf_gain_function
        self.max_nodes = max_nodes
        self.max_depth = max_depth

    def get_tags(self):
        return {s.algorithm: "DecisionTree",
                "Used": self.used_percentage,
                "MinSamples": self.min_samples,
                "MaxDepth": self.max_depth,
                "MaxNodes": self.max_nodes,
                "InfGain": self.inf_gain_function.name}

    def train(self, dataset: Dataset):
        used_ds = dataset
        if self.used_percentage != 1:
            used_ds = dataset.build_random_sample_dataset(self.used_percentage)

        self.data_frame = used_ds.getRows().copy()
        self.categories = used_ds.getAttributes().copy()
        self.class_col = used_ds.getClassAttr()
        self.tree = self.evaluateNextNode(self.data_frame, self.categories, self.class_col, 0, self.max_nodes)
        self.is_trained = True
        print(self.tree)

    def evaluateNextNode(self, data_frame: pd.DataFrame, categories, class_col, depth, max_nodes):
        class_col_freq = get_class_probs(data_frame, class_col)
        most_likely_class = class_col_freq.idxmax()
        head = class_col_freq.head().index.values
        if len(head) == 1 or len(categories) == 0 or self.is_dataset_too_small(data_frame) or self.is_branch_too_deep(depth)\
                or self.has_only_one_node_remaining(max_nodes):
            # This means there are different possible classes but there are no more categories to expand
            return Tree(Node.fromClazz(str(most_likely_class), class_col_freq, len(data_frame)))
        else:
            categoryData = pd.DataFrame(data=0, index=categories, columns=["InfGain"])
            categories = categories.copy()
            for category in categories:
                categoryData.loc[category, "InfGain"] = self.inf_gain_function.value.func(data_frame, category, class_col)
            most_informational_category = categoryData["InfGain"].idxmax()
            if categoryData["InfGain"][most_informational_category] <= 0.0001:
                categories.remove(most_informational_category)
                return Tree(Node.fromClazz(str(most_likely_class), class_col_freq, len(data_frame)))
            possibleValues = data_frame.groupby(most_informational_category).agg(count=(class_col, 'count')).index.values
            categories.remove(most_informational_category)
            leafs = []
            remaining_nodes = self.substract_if_not_none(max_nodes)
            for possibleValue in possibleValues:
                if remaining_nodes is None or remaining_nodes > 0:
                    cut_data_frame = data_frame[data_frame[most_informational_category] == possibleValue]
                    leafs.append(Tree(Node.fromValue(possibleValue, get_class_probs(cut_data_frame, class_col), len(cut_data_frame)),
                                      self.evaluateNextNode(cut_data_frame, categories, class_col, depth=depth+1, max_nodes=remaining_nodes)))
                    remaining_nodes = self.substract_if_not_none(remaining_nodes)
            return Tree(Node.fromAttr(most_informational_category, class_col_freq, len(data_frame)), leafs)

    def get_tree(self):
        return self.tree

    def substract_if_not_none(self, max_nodes):
        if max_nodes is not None:
            return max_nodes - 1
        else:
            return None

    def is_dataset_too_small(self, data_frame:pd.DataFrame):
        return self.min_samples is not None and len(data_frame) <= self.min_samples

    def is_branch_too_deep(self, depth):
        return self.max_depth is not None and depth >= self.max_depth

    def has_only_one_node_remaining(self, max_nodes):
        return max_nodes is not None and max_nodes == 1

    def evaluate(self, values_dict):
        node = self.tree
        prevNode = 0
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



