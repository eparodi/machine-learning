import os, sys; sys.path.append(os.getcwd())

from common.algorithms.knn import KNN
from common.dataset.dataset_builder import create_feelings_dataset
from common.experiment.experiment_comparer import Comparer, TestType
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

dataset = create_feelings_dataset()
print('Average wordcount and sentimentValue:\n', dataset.getRows().groupby('StarRating').mean())

algorithms = []
algorithms.append(KNN())
algorithms.append(KNN(weighted=True))

for x in range(3, 10):
    comparer = Comparer(dataset, x/10, algorithms, test_type=TestType.DISJOINT)
    print(comparer)
    for r in comparer.results:
        print(r.confusion)

df = dataset.getRows()
df['StarRating'] = df['StarRating'].replace({
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "brown",
})
threedee = plt.figure().gca(projection='3d')
threedee.scatter(df['wordcount'], df['titleSentiment'], df['sentimentValue'], c=df['StarRating'])
threedee.set_xlabel('wordcount')
threedee.set_ylabel('titleSentiment')
threedee.set_zlabel('sentimentValue')
plt.show()
