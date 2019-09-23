from .algorithm import Algorithm
from common.dataset.dataset import Dataset

import numpy as np

class KNN(Algorithm):

    def __init__(self, tr_perc=0.5):
        super().__init__()
        self.tr_perc = tr_perc

    def train(self, dataset: Dataset):
        rows = dataset.build_random_sample_dataset(self.tr_perc).getRows()
        self.rows = rows.replace(to_replace={'negative': -1, 'positive': 1, np.nan: 0})

    def __weighted_evaluation(self, weights, k):
        sums = weights.head(k).groupby('Star Rating').sum().sort_values('dist')
        return sums.idxmin()['dist']

    def __unweighted_evaluation(self, weights, k):
        result = None
        while not result:
            sums = weights.head(k).groupby(
                'Star Rating').count().sort_values('dist', ascending=False)
            values = list(sums.values[:, 0])
            if values.count(max(values)) == 1:
                return sums.idxmax()['dist']
            k += 1

    def evaluate(self, values_dict, k=5, weighted=False):
        distances = self.rows
        wordCount = values_dict['wordcount']
        distances['dist'] = np.sqrt(
            (distances['wordcount'] - values_dict['wordcount']) ** 2 +
            (distances['titleSentiment'] - values_dict['titleSentiment']) ** 2 +
            (distances['sentimentValue'] - values_dict['sentimentValue']) ** 2)
        distances = distances[['dist', 'Star Rating']]
        distances = distances.sort_values(by=['dist'])
        if weighted:
            return self.__weighted_evaluation(distances, k)
        else:
            return self.__unweighted_evaluation(distances, k)

    def get_tags(self):
        return super().get_tags()
