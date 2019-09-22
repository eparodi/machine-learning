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
    def __init__(self, clazz_attr, rows=None, dataset_path=None, dataset_type=None, blacklisted_attrs=(), attr_generators=(),dataset=None, numerify=False):
        if (rows is None or dataset is None) and (dataset_path is None or dataset_type is None):
            raise AssertionError("You must init with (rows and dataset) or with (dataset_path and dataset_type)!")
        if (rows is not None or dataset is not None) and (dataset_path is not None or dataset_type is not None):
            raise AssertionError("You must init with (rows and dataset) OR with (dataset_path and dataset_type) but not both!")
        self.clazz_attr = clazz_attr

        if rows is not None and dataset is not None:
            self.rows = rows
            self.clazz_attr_values = dataset.getClassAttrValues().copy()
            self.attributes = dataset.getAttributes()
        elif dataset_path is not None and dataset_type is not None:
            orig_rows = Dataset.loadRows(dataset_path, dataset_type)
            blacklisted_attrs = Dataset.loadBlacklists(blacklisted_attrs)
            attr_generators = attr_generators
            all_rows = Dataset.generateAttrs(orig_rows, attr_generators)
            self.rows = Dataset.loadFilteredRows(all_rows, blacklisted_attrs)

            all_attributes = list(all_rows)
            attributes_with_clazz = Dataset.loadFilteredAttrs(all_attributes, blacklisted_attrs)
            self.attributes = [x for x in attributes_with_clazz if x != self.clazz_attr]
            self.clazz_attr_values = Dataset.loadAttrValues(self.rows, self.clazz_attr)
        else:
            raise AssertionError("Something went wrong with dataset creation")

        # self.numerify = numerify
        # if numerify:
        #     self.numericToCategoric = Dataset.loadNumericToCategoricMap(self.rows, self.all_attributes)

    def build_random_sample_dataset(self, frac):
        sampled_rows = self.getRows().sample(frac=frac, replace=True)
        return self.copy_dataset_with_new_rows(sampled_rows)

    def split_dataset(self, frac):
        shuffled_rows = self.getRows().sample(frac=1)
        split_index = int(frac*len(shuffled_rows))
        return self.copy_dataset_with_new_rows(shuffled_rows[:split_index]), \
               self.copy_dataset_with_new_rows(shuffled_rows[split_index:])

    def copy_dataset_with_new_rows(self, rows):
        return Dataset(self.clazz_attr, rows=rows, dataset=self)

    def getClassAttr(self):
        return self.clazz_attr

    def getRows(self):
        return self.rows

    def getAttributes(self):
        return self.attributes

    def getClassAttrValues(self):
        return self.clazz_attr_values

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
    def loadRows(dataset_path, dataset_type):
        base_path = dataset_folder_path
        if not os.path.exists(base_path + dataset_path):
            base_path = alt_dataset_folder_path

        if dataset_type == Dataset.Type.CSV:
            return pd.read_csv(base_path + dataset_path, sep=',', header=0)
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
