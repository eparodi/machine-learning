from enum import Enum

import pandas as pd

dataset_folder_path = "../datasets/"

class Dataset():
    class Type(Enum):
        EXCEL = 1
        TSV = 2
        CSV = 3

    def __init__(self, dataset_path, clazz_attr, blacklisted_attrs, dataset_type, attr_generators = ()):
        assert isinstance(dataset_path, str), "dataset_path not a str!: " + str(dataset_path.__class__)
        self.dataset_path = dataset_path
        assert isinstance(clazz_attr, str), "clazz_attr not a str!: " + str(clazz_attr.__class__)
        self.clazz_attr = clazz_attr
        self.dataset_type = dataset_type
        self.blacklisted_attrs = Dataset.loadBlacklists(blacklisted_attrs)
        self.orig_rows = Dataset.loadRows(dataset_path, dataset_type)
        self.all_rows = Dataset.generateAttrs(self.orig_rows, attr_generators)
        self.rows = Dataset.loadFilteredRows(self.all_rows, self.blacklisted_attrs)
        self.all_attributes = list(self.all_rows)
        self.attributes_with_clazz = Dataset.loadFilteredAttrs(self.all_attributes, self.blacklisted_attrs)
        self.attributes = [x for x in self.attributes_with_clazz if x != self.clazz_attr]

    def getClassAttr(self):
        return self.clazz_attr

    def getRows(self):
        return self.rows

    def getAllRows(self):
        return self.all_rows

    def getAttributes(self):
        return self.attributes

    def getAttributesWithClass(self):
        return self.attributes_with_clazz

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
        if dataset_type == Dataset.Type.CSV:
            return pd.read_csv(dataset_folder_path + dataset_path, sep=',', header=0)
        elif dataset_type == Dataset.Type.TSV:
            return pd.read_csv(dataset_folder_path + dataset_path, sep='\t', header=0)
        elif dataset_type == Dataset.Type.EXCEL:
            return pd.ExcelFile(dataset_folder_path + dataset_path).parse()
        else:
            raise AssertionError("Dataset type not supported!: " + str(dataset_type))

    @staticmethod
    def create_tenis_dataset():
        return Dataset("juegaTenis.csv", "Juega", ["Dia"], Dataset.Type.CSV)

    @staticmethod
    def create_rio_dataset():
        return Dataset("disfrutaRio.csv", "Disfruta", ["id"], Dataset.Type.CSV)

    @staticmethod
    def create_students_dataset():
        return Dataset("binary.csv", "admit", [], Dataset.Type.CSV)

    @staticmethod
    def create_britons_dataset():
        return Dataset("britons.xlsx", "Nacionalidad", [], Dataset.Type.EXCEL)

    @staticmethod
    def create_news_dataset():
        return Dataset("news.tsv", "categoria", ["fecha"], Dataset.Type.TSV)

    @staticmethod
    def bucketed_age(age):
        if age < 12:
            return "Child"
        elif age < 16:
            return "Preteen"
        elif age < 22:
            return "YoungAdult"
        elif age < 60:
            return "Adult"
        else:
            return "Old"

    @staticmethod
    def create_titanic_dataset():
        blacklistedAttrs = ["PassengerId", "Name", "Cabin", "Ticket", "SibSp", "Parch", "Fare", "Embarked"]
        gen = [ ("Age", lambda r: r["Age"].apply(lambda x: Dataset.bucketed_age(x))),
                ("Survived", lambda r: r["Survived"].apply(lambda x: ["Dead", "Survivor"][x]))]
        return Dataset("titanic.csv", "Survived", blacklistedAttrs, Dataset.Type.TSV, attr_generators=gen)

    @staticmethod
    def generateAttrs(orig_rows, attr_generators):
        all_rows = orig_rows.copy()
        for attr_gen in attr_generators:
            attr_name = attr_gen[0]
            attr_formula = attr_gen[1]
            all_rows[attr_name] = attr_formula(all_rows)
        return all_rows

