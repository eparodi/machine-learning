import os, sys; sys.path.append(os.getcwd())

from common.algorithms.knn import KNN
from common.dataset.dataset_builder import create_feelings_dataset

dataset = create_feelings_dataset()
print('Average wordcount and sentimentValue:\n', dataset.getRows().groupby('Star Rating').mean())
knn = KNN()
knn.train(dataset)