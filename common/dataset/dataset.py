from enum import Enum, auto
import os.path
import pandas as pd

dataset_folder_path = "../datasets/"
alt_dataset_folder_path = "../../datasets/"


class Dataset:
    class Type(Enum):
        EXCEL = auto()
        TSV = auto()
        CSV = auto()

    # Hacer que transforme las variables categoricas en numericas
    # Hacer que agrupe los valores en cada atributo y les asigne un numero de orden
    def __init__(self, clazz_attr, rows, attributes, clazz_attr_values):
        self.clazz_attr = clazz_attr
        self.rows=rows
        self.attributes=attributes
        self.clazz_attr_values=clazz_attr_values

    @staticmethod
    def build_dataset_from_path(clazz_attr, dataset_path, dataset_type, sep=",", blacklisted_attrs=(), attr_generators=(), remove_nan=False, normalize=False):
        orig_rows = Dataset.loadRows(dataset_path, dataset_type, sep)
        blacklisted_attrs = Dataset.loadBlacklists(blacklisted_attrs)
        attr_generators = attr_generators
        all_rows = Dataset.generateAttrs(orig_rows, attr_generators)
        rows = Dataset.loadFilteredRows(all_rows, blacklisted_attrs)

        all_attributes = list(all_rows)
        attributes_with_clazz = Dataset.loadFilteredAttrs(all_attributes, blacklisted_attrs)
        attributes = [x for x in attributes_with_clazz if x != clazz_attr]
        clazz_attr_values = Dataset.loadAttrValues(rows, clazz_attr)
        if remove_nan:
            rows = rows.dropna()
        if normalize:
            rows = Dataset.normalize_data(rows)
        return Dataset(clazz_attr=clazz_attr, rows=rows, attributes=attributes, clazz_attr_values=clazz_attr_values)

    @staticmethod
    def build_dataset_from_dataset(dataset, rows):
        return Dataset(clazz_attr=dataset.getClassAttr(), rows=rows, attributes=dataset.getAttributes(), clazz_attr_values=dataset.getClassAttrValues())

    @staticmethod
    def build_dataset_from_rows(clazz_attr, rows, blacklisted_attrs=()):
        blacklisted_attrs = Dataset.loadBlacklists(blacklisted_attrs)
        all_attributes = list(rows)
        attributes_with_clazz = Dataset.loadFilteredAttrs(all_attributes, blacklisted_attrs)
        attributes = [x for x in attributes_with_clazz if x != clazz_attr]
        clazz_attr_values = Dataset.loadAttrValues(rows, clazz_attr)
        return Dataset(clazz_attr=clazz_attr, rows=rows, attributes=attributes, clazz_attr_values=clazz_attr_values)

    def build_random_sample_dataset(self, frac, replace=True):
        sampled_rows = self.getRows().sample(frac=frac, replace=replace)
        return self.copy_dataset_with_new_rows(sampled_rows)

    def split_dataset(self, frac):
        shuffled_rows = self.getRows().sample(frac=1)
        split_index = int(frac*len(shuffled_rows))
        return self.copy_dataset_with_new_rows(shuffled_rows[:split_index]), \
               self.copy_dataset_with_new_rows(shuffled_rows[split_index:])

    def partition_dataset(self, partitions_amount, partition_index):
        partition_size = int(len(self.getRows())/partitions_amount)
        start = partition_index*partitions_amount
        test_set = self.copy_dataset_with_new_rows(self.getRows()[start:start + partition_size])
        training_sets = []
        if start != 0:
            training_sets.append(self.getRows()[0:start-1])
        if start + partition_size < len(self.getRows()):
            training_sets.append(self.getRows()[start + partition_size+1:-1])
        return self.copy_dataset_with_new_rows(pd.concat(training_sets)), test_set

    def copy_dataset_with_new_rows(self, rows):
        return Dataset.build_dataset_from_dataset(rows=rows, dataset=self)

    def getClassAttr(self):
        return self.clazz_attr

    def getRows(self):
        return self.rows

    def getAttributes(self):
        return self.attributes

    def getClassAttrValues(self):
        return self.clazz_attr_values

    @staticmethod
    def normalize_data(rows):
        for col in rows.columns:
            rows[col] -= rows[col].min()
            rows[col] /= rows[col].max()
        return rows

    # def get_numeric_to_categoric(self):
    #     return self.numericToCategoric

    def __str__(self):
        return str(self.getRows())

    # @staticmethod
    # def categoricToNumeric(rows, numericToCategoric):
    #     filtered_rows = all_rows.copy()
    #     for attr in blacklisted_attrs:
    #         filtered_rows = filtered_rows.drop(attr, axis=1)
    #     return filtered_rows

    @staticmethod
    def loadFilteredRows(all_rows, blacklisted_attrs):
        filtered_rows = all_rows.copy()
        for attr in blacklisted_attrs:
            filtered_rows = filtered_rows.drop(attr, axis=1)
        return filtered_rows

    @staticmethod
    def loadFilteredAttrs(all_attributes, blacklisted_attrs):
        return [x for x in all_attributes if x not in blacklisted_attrs]

    @staticmethod
    def loadBlacklists(blacklisted_attrs):
        if isinstance(blacklisted_attrs, tuple) or isinstance(blacklisted_attrs, list):
            for attr in blacklisted_attrs:
                assert isinstance(attr, str), "Blacklist attr not a str!: " + str(attr.__class__)
            return blacklisted_attrs
        elif isinstance(blacklisted_attrs, str):
            return [blacklisted_attrs]
        else:
            raise AssertionError("blacklisted_attrs not a str or a list or a tuple!: " + str(blacklisted_attrs.__class__))

    @staticmethod
    def loadRows(dataset_path, dataset_type, sep=','):
        base_path = dataset_folder_path
        if not os.path.exists(base_path + dataset_path):
            base_path = alt_dataset_folder_path

        if dataset_type == Dataset.Type.CSV:
            return pd.read_csv(base_path + dataset_path, sep=sep, header=0)
        elif dataset_type == Dataset.Type.TSV:
            return pd.read_csv(base_path + dataset_path, sep='\t', header=0)
        elif dataset_type == Dataset.Type.EXCEL:
            return pd.ExcelFile(base_path + dataset_path).parse()
        else:
            raise AssertionError("Dataset type not supported!: " + str(dataset_type))

    @staticmethod
    def loadAttrValues(rows, attr):
        return rows[attr].unique()    \

    @staticmethod
    def loadNumericToCategoricMap(rows, attrs):
        numericToCategoricMap = {}
        for attr in attrs:
            numericToCategoricMap[attr] = Dataset.loadAttrValues(rows, attr)
        return numericToCategoricMap

    @staticmethod
    def generateAttrs(orig_rows, attr_generators):
        all_rows = orig_rows.copy()
        for attr_gen in attr_generators:
            attr_name = attr_gen[0]
            attr_formula = attr_gen[1]
            all_rows[attr_name] = attr_formula(all_rows)
        return all_rows
