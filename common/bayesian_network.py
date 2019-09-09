from pandas import DataFrame

class BayesianNetwork:

    def __init__(self, graph: dict, tables: dict, reversed_graph: dict = None):
        self.graph = graph
        self.tables = tables
        self.reversed_graph = reversed_graph if reversed_graph else BayesianNetwork.__reverse_graph(graph)

    @staticmethod
    def __reverse_graph(graph: dict):
        reversed_graph = {key: set() for key in graph.keys()}
        for key in graph.keys():
            reversed_graph[key].add(key)
            for value in graph[key]:
                reversed_graph[value].add(key)

        # This calculates the clause of the graph.
        change = True
        while change:
            change = False
            for key in reversed_graph.keys():
                for value in reversed_graph[key]:
                    current_len = len(reversed_graph[value])
                    for c in reversed_graph[value]:
                        reversed_graph[value].add(c)
                    change = True if current_len != len(reversed_graph[value]) or change else False

        return reversed_graph

    @staticmethod
    def __generate_tables(reversed_graph: dict, data_frame: DataFrame):
        tables = dict()
        for key in reversed_graph.keys():
            columns = list(reversed_graph[key])
            tables[key] = data_frame.groupby(columns).agg({key: ['count']})
            tables[key] = tables[key].div(tables[key].sum()).to_dict()[
                list(tables[key].keys())[0]]
        return tables

    @staticmethod
    def from_data_frame(graph: dict, data_frame: DataFrame):
        reversed_graph = BayesianNetwork.__reverse_graph(graph)
        tables = BayesianNetwork.__generate_tables(reversed_graph, data_frame)
        return BayesianNetwork(graph, tables, reversed_graph)

    def __get_table_key(self, key, inp, keys):
        length = len(self.reversed_graph[key])
        if length == 1:
            return inp[keys.index(key)]
        else:
            return tuple(inp[keys.index(k)] for k in self.reversed_graph[key])

    def calculate_probability(self, inp):
        keys = [key for key in self.graph.keys()]
        result = 1
        for key in keys:
            table_key = self.__get_table_key(key, inp, keys)
            if (not table_key in self.tables[key]):
                result *= 0
            else:
                result *= self.tables[key][table_key]
        return result

    def calculate_event_probability(self, key, value):
        keys = self.reversed_graph[key]
        probs = self.tables[key]
        index = list(keys).index(key)
        fprob = 0
        if len(self.reversed_graph[key]) > 1:
            for prob in probs.keys():
                if prob[index] == value:
                    fprob += probs[prob]
        else:
            fprob += probs[value]
        return fprob

    def __calculate_joint_probability(self, inp):
        of_key, of_value = inp['of']
        if_key, if_value = inp['if']
        if (not if_key in self.reversed_graph[of_key] and
            not of_key in self.reversed_graph[if_key]):
            return (self.calculate_event_probability(of_key, of_value) *
                self.calculate_event_probability(if_key, if_value))
        if if_key in self.reversed_graph[of_key]:
            probs = self.tables[of_key]
            keys = list(self.reversed_graph[of_key])
        else:
            probs = self.tables[if_key]
            keys = list(self.reversed_graph[if_key])
        if_index = keys.index(if_key)
        of_index = keys.index(of_key)
        fprob = 0
        for prob in probs.keys():
            if prob[if_index] == if_value and prob[of_index] == of_value:
                fprob += probs[prob]
        return fprob

    def calculate_conditional_probability(self, inp):
        if_prob = self.calculate_event_probability(inp['of'][0], inp['of'][1])
        joint_prob = self.__calculate_joint_probability(inp)
        return joint_prob / if_prob
