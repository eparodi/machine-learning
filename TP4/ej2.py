import os, sys; sys.path.append(os.getcwd())

from common.dataset.journalist_dataset import create_journalist_dataset
from common.algorithms.hierarchical_clustering import HierarchicalClustering
from common.algorithms.knearest import KNearest

def get_journalists(data):
    journalists = [d["journalist"] for d in data]
    j_dict = dict.fromkeys(journalists)
    for j in j_dict.keys():
        j_dict[j] = journalists.count(j)
    print(j_dict)

dataset = create_journalist_dataset()

for method in HierarchicalClustering.METHODS:
    alg = HierarchicalClustering(method)
    print(alg.get_tags())
    alg.train(dataset)
    groups = alg.get_groups(5)
    for group in groups:
        data = group["data"]
        get_journalists(data)

knearest = KNearest(5)
print(knearest.get_tags())
knearest.train(dataset)
data = knearest.data_frame
for i in range(0, 5):
    klass = data[data[knearest.CLASS_KEY] == i]
    klass = [row[1] for row in klass.iterrows()]
    get_journalists(klass)
