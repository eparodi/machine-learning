from common.naive_bayes import NaiveBayes
import common.dataset.dataset_builder as db


ds = db.create_britons_dataset()
britons = ds.getRows()
inp = (1, 0, 1, 1, 0)
bayes = NaiveBayes.from_data_frame(britons, "Nacionalidad")
print(bayes.get_probabilities(inp))

