import random

import pandas as pd

from common.dataset.dataset import Dataset
import numpy as np

def create_tenis_dataset():
    return Dataset.build_dataset_from_path(clazz_attr="Juega", blacklisted_attrs=["Dia"], dataset_path="juegaTenis.csv", dataset_type=Dataset.Type.CSV)


def create_rio_dataset():
    return Dataset.build_dataset_from_path(clazz_attr="Disfruta", blacklisted_attrs=["id"], dataset_path="disfrutaRio.csv", dataset_type=Dataset.Type.CSV)

def bucketed_gpa(gpa):
    if gpa < 1:
        return "<1"
    elif gpa < 2:
        return "<2"
    elif gpa < 3:
        return "<3"
    elif gpa < 3.5:
        return "<3.5"
    elif gpa < 3.75:
        return "<3.75"
    else:
        return "<4"

def bucketed_gre(gre):
    if gre < 200:
        return "<200"
    elif gre < 400:
        return "<400"
    elif gre < 600:
        return "<600"
    elif gre < 800:
        return "<800"
    elif gre < 900:
        return "<900"
    else:
        return "<1000"

def create_students_dataset(bucket=False):
    if bucket:
        gen = [("gpa", lambda r: r["gpa"].apply(lambda x: bucketed_gpa(x))),
           ("gre", lambda r: r["gre"].apply(lambda x: bucketed_gre(x)))]
    else:
        gen = []
    return Dataset.build_dataset_from_path(clazz_attr="admit", dataset_path="binary.csv", dataset_type=Dataset.Type.CSV, attr_generators=gen)


def create_britons_dataset():
    return Dataset.build_dataset_from_path(clazz_attr="Nacionalidad", dataset_path="britons.xlsx", dataset_type=Dataset.Type.EXCEL)


def create_news_dataset():
    return Dataset.build_dataset_from_path(clazz_attr="categoria", blacklisted_attrs=["fecha", "fuente"], dataset_path="news.tsv", dataset_type=Dataset.Type.TSV)

def create_feelings_dataset():
    string_values = {
        'negative': 0,
        'positive': 1,
        np.nan: 0.5
    }
    gen = [
        ("titleSentiment", lambda r: r["titleSentiment"].apply(lambda x: string_values[x])),
        ("wordcount", lambda r: r["wordcount"] / r["wordcount"].max()),
    ]
    return Dataset.build_dataset_from_path(
        clazz_attr="StarRating",
        blacklisted_attrs=["Review Title", "Review Text", "textSentiment"],
        dataset_path="reviews_sentiment.csv",
        dataset_type=Dataset.Type.CSV,
        sep=";",
        attr_generators=gen)

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


def create_titanic_dataset():
    blacklistedAttrs = ["PassengerId", "Name", "Cabin", "Ticket", "Parch", "Fare", "SibSp", "Embarked"]
    gen = [("Age", lambda r: r["Age"].apply(lambda x: bucketed_age(x))),
           ("Survived", lambda r: r["Survived"].apply(lambda x: ["Dead", "Survivor"][x]))]
    return Dataset.build_dataset_from_path(clazz_attr="Survived", blacklisted_attrs=blacklistedAttrs, dataset_path="titanic.csv", dataset_type=Dataset.Type.TSV, attr_generators=gen)


def create_linearly_separable_dataset(n=20, height=100, width=100, center=0.2):
    x1 = random.uniform(-(width * center) / 2, (width * center) / 2)
    x2 = random.uniform(-(width * center) / 2, (width * center) / 2)
    y1 = random.uniform(-(height * center) / 2, (height * center) / 2)
    y2 = random.uniform(-(height * center) / 2, (height * center) / 2)

    def point(x, y):
        return {"x": x, "y": y}

    def pointInLine(x):
        return y1 + (y2 - y1) / (x2 - x1) * (x - x1)

    colors = ["r", "g"]
    row_list = []
    for x in range(0, n):
        data = {}
        data["x"] = random.uniform(-width / 2, width / 2)
        data["y"] = random.uniform(-height / 2, height / 2)
        d = (data["x"] - x1) * (y2 - y1) - (data["y"] - y1) * (x2 - x1)
        data["tag"] = 0 if d <= 0 else 1
        data["color"] = colors[data["tag"]]
        row_list.append(data)

    df = pd.DataFrame(row_list, columns=["tag", "x", "y", "color"])
    return Dataset.build_dataset_from_rows(rows=df, clazz_attr="tag", blacklisted_attrs="color")

def create_heart_dataset(blacklisted=[]):
    blacklistedAttrs = ["tvdlm"] + blacklisted
    return Dataset.build_dataset_from_path("sigdz", blacklisted_attrs=blacklistedAttrs,
        dataset_path="acath.xls", dataset_type=Dataset.Type.EXCEL, remove_nan=True, normalize=True)
