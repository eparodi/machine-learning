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
            # if len(reversed_graph[key]) != 1:
            #     tables[key] = data_frame.groupby(
            #         list(reversed_graph[key])).mean()[key].to_dict()
            # else:
            # tables[key] = data_frame.groupby(key).agg({key: ['count']})
            # tables[key] = tables[key].div(tables[key].sum()).to_dict()[
            #     list(tables[key].keys())[0]]
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


