import os, sys; sys.path.append(os.getcwd())

from common.dataset.journalist_dataset import create_journalist_dataset
from common.algorithms.hierarchical_clustering import HierarchicalClustering
from common.algorithms.knearest import KNearest

def get_journalists(data):
    journalists = [d["journalist"] for d in data]
    j_dict = dict.fromkeys(journalists)
    for j in j_dict.keys():
        print("{journalist}: {count}".format(journalist=j, count=journalists.count(j)))
        j_dict[j] = journalists.count(j)
    return j_dict

def remove_from_group(group, dataset):
    for data in group["data"]:
        dataset.rows = dataset.rows[dataset.rows["filename"] != data["filename"]]

dataset = create_journalist_dataset(blacklisted=[
    # "vocabulary_extension",
    # "average_sentence_length",
    # "coordinant_numbers",
    "indeterminant",
    "determinant",
    "mente_adverbs",
    "most_used_words",
])
rows = dataset.rows.copy()

for method in HierarchicalClustering.METHODS:
    dataset.rows = rows.copy()
    repeat = True
    while repeat:
        repeat = False
        alg = HierarchicalClustering(method)
        print(alg.get_tags())
        alg.train(dataset)
        groups = alg.get_groups(5)
        i = 1
        for group in groups:
            print("---Group {n}---".format(n=i))
            data = group["data"]
            i += 1
            get_journalists(data)
            if len(group["data"]) <= 3:
                remove_from_group(group, dataset)
                repeat = True


knearest = KNearest(5)
print(knearest.get_tags())
knearest.train(dataset)
data = knearest.data_frame
for i in range(0, 5):
    print("---Group {n}---".format(n=i + 1))
    klass = data[data[knearest.CLASS_KEY] == i]
    klass = [row[1] for row in klass.iterrows()]
    get_journalists(klass)
