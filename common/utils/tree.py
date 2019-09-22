from pandas import DataFrame
from typing import List

class Tree():
    def __init__(self, value, leafs=()):
        self.value = value
        if isinstance(leafs, tuple) or isinstance(leafs, list):
            for leaf in leafs:
                if not isinstance(leaf, Tree):
                    raise AssertionError("Not a list leaf!: " + str(leafs.__class__))
            self.leafs = leafs
        elif isinstance(leafs, Tree):
            self.leafs = [leafs]
        else:
            raise AssertionError("Not a leaf!: " + str(leafs.__class__))

    def __addChild__(self, leaf):
        if isinstance(leaf, Tree):
            self.leafs.append(leaf)
        else:
            raise AssertionError("Not a leaf!: " + str(leaf.__class__))

    def showTabbed(self, tab):
        val = str(self.value) + "\n"
        last = len(self.leafs)
        for x in range(last):
            leaf = self.leafs[x]
            if x == last -1:
                val = val + tab + "╚═>" + leaf.showTabbed(tab + "   ")
            else:
                val = val + tab + "╠═>" + leaf.showTabbed(tab + "║  ")
        return val

    def __str__(self):
        return self.showTabbed("")

    def bottomLeafValues(self):
        values = []
        if len(self.leafs) == 0:
            values = [self.value]
        else:
            for leaf in self.leafs:
                values.extend(leaf.bottomLeafValues())
        return values
