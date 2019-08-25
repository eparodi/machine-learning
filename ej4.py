from bayesian_network import BayesianNetwork

import pandas as pd

data = pd.read_csv('binary.csv', sep=',', header=0)
data['gre'] = data['gre'].ge(500)
data['gpa'] = data['gpa'].ge(3)

graph = {
    'rank': ['gre', 'gpa', 'admit'],
    'gre': ['admit'],
    'gpa': ['admit'],
    'admit': [],
}

bayes_net = BayesianNetwork.from_data_frame(graph, data)

print([key for key in bayes_net.graph.keys()])

print(bayes_net.calculate_probability((1, False, False, 1)))
print(bayes_net.calculate_probability((2, False, False, 1)))
print(bayes_net.calculate_probability((3, False, False, 1)))
print(bayes_net.calculate_probability((4, False, False, 1)))

print(bayes_net.calculate_probability((1, False, False, 0)))
print(bayes_net.calculate_probability((2, False, False, 0)))
print(bayes_net.calculate_probability((3, False, False, 0)))
print(bayes_net.calculate_probability((4, False, False, 0)))

print(bayes_net.calculate_probability((1, True, False, 1)))
print(bayes_net.calculate_probability((2, True, False, 1)))
print(bayes_net.calculate_probability((3, True, False, 1)))
print(bayes_net.calculate_probability((4, True, False, 1)))

print(bayes_net.calculate_probability((1, True, False, 0)))
print(bayes_net.calculate_probability((2, True, False, 0)))
print(bayes_net.calculate_probability((3, True, False, 0)))
print(bayes_net.calculate_probability((4, True, False, 0)))

print(bayes_net.calculate_probability((1, True, True, 0)))
print(bayes_net.calculate_probability((2, True, True, 0)))
print(bayes_net.calculate_probability((3, True, True, 0)))
print(bayes_net.calculate_probability((4, True, True, 0)))

print(bayes_net.calculate_probability((1, True, True, 1)))
print(bayes_net.calculate_probability((2, True, True, 1)))
print(bayes_net.calculate_probability((3, True, True, 1)))
print(bayes_net.calculate_probability((4, True, True, 1)))
