from common.dataset.dataset import Dataset


def create_tenis_dataset():
    return Dataset("Juega", blacklisted_attrs=["Dia"], dataset_path="juegaTenis.csv", dataset_type=Dataset.Type.CSV)


def create_rio_dataset():
    return Dataset("Disfruta", blacklisted_attrs=["id"], dataset_path="disfrutaRio.csv", dataset_type=Dataset.Type.CSV)


def create_students_dataset():
    return Dataset("admit", dataset_path="binary.csv", dataset_type=Dataset.Type.CSV)


def create_britons_dataset():
    return Dataset("Nacionalidad", dataset_path="britons.xlsx", dataset_type=Dataset.Type.EXCEL)


def create_news_dataset():
    return Dataset("categoria", blacklisted_attrs=["fecha", "fuente"], dataset_path="news.tsv", dataset_type=Dataset.Type.TSV)

def create_feelings_dataset():
    return Dataset("Star Rating", blacklisted_attrs=["Review Title", "Review Text", "textSentiment"], dataset_path="reviews_sentiment.csv", dataset_type=Dataset.Type.CSV, sep=";")

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
