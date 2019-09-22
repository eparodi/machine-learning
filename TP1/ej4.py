import pandas as pd

from common.bayesian_network import BayesianNetwork
import common.dataset.dataset_builder as db

data = db.create_students_dataset().getRows()
data['gre'] = data['gre'].ge(500)
data['gpa'] = data['gpa'].ge(3)

graph = {
    'rank': ['gre', 'gpa', 'admit'],
    'gre': ['admit'],
    'gpa': ['admit'],
    'admit': [],
}

bayes_net = BayesianNetwork.from_data_frame(graph, data)

print('Probability of school rank 2, GRE < 500, GPA > 3 and admitted.')
print(bayes_net.calculate_probability((2, False, True, 1)))
print('Probability of being admitted if the school was rank 1.')
print(bayes_net.calculate_conditional_probability(
    {
        'of': ('admit', 0),
        'if': ('rank', 1),
    }
))
