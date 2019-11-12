import os, sys; sys.path.append(os.getcwd())

from common.dataset.journalist_dataset import create_journalist_dataset
from common.algorithms.hierarchical_clustering import HierarchicalClustering

alg = HierarchicalClustering(HierarchicalClustering.SINGLE_LINK)
dataset = create_journalist_dataset()
alg.train(dataset)
groups = alg.get_groups(5)
for group in groups:
    data = group["data"]
    journalists = [d["journalist"] for d in data]
    j_dict = dict.fromkeys(journalists)
    for j in j_dict.keys():
        j_dict[j] = journalists.count(j)
    print(j_dict)
