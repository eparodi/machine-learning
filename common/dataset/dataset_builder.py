from common.dataset.dataset import Dataset
import numpy as np

def create_tenis_dataset():
    return Dataset("Juega", blacklisted_attrs=["Dia"], dataset_path="juegaTenis.csv", dataset_type=Dataset.Type.CSV)


def create_rio_dataset():
    return Dataset("Disfruta", blacklisted_attrs=["id"], dataset_path="disfrutaRio.csv", dataset_type=Dataset.Type.CSV)

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
    return Dataset("admit", dataset_path="binary.csv", dataset_type=Dataset.Type.CSV, attr_generators=gen)


def create_britons_dataset():
    return Dataset("Nacionalidad", dataset_path="britons.xlsx", dataset_type=Dataset.Type.EXCEL)


def create_news_dataset():
    return Dataset("categoria", blacklisted_attrs=["fecha", "fuente"], dataset_path="news.tsv", dataset_type=Dataset.Type.TSV)

def create_feelings_dataset():
    string_values = {
        'negative': 0,
        'positive': 1,
        np.nan: 0.5
    }
    gen = [
        ("titleSentiment", lambda r: r["titleSentiment"].apply(lambda x: string_values[x])),
        ("wordcount", lambda r: r["wordcount"] / r["wordcount"].max()),
        ("sentimentValue", lambda r: r["sentimentValue"] / r["sentimentValue"].max())
    ]
    return Dataset(
        "StarRating",
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
    blacklistedAttrs = ["PassengerId", "Name", "Cabin", "Ticket", "Parch", "Fare"]
    gen = [("Age", lambda r: r["Age"].apply(lambda x: bucketed_age(x))),
           ("Survived", lambda r: r["Survived"].apply(lambda x: ["Dead", "Survivor"][x]))]
    return Dataset("Survived", blacklisted_attrs=blacklistedAttrs, dataset_path="titanic.csv", dataset_type=Dataset.Type.TSV, attr_generators=gen)
