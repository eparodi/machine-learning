import matplotlib.pyplot as plt
import os, sys; sys.path.append(os.getcwd())

from common.algorithms.svm import SVM
from common.experiment.experiment_comparer import TestType, Comparer
import common.dataset.dataset_builder as db

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_decision_regions

# ds = db.create_heart_dataset()
# algorithms = []
# algorithms.append(SVM(kernel='linear'))
# # algorithms.append(SVM(kernel='polynomial'))
# algorithms.append(SVM(kernel='rbf'))
# algorithms.append(SVM(kernel='sigmoid'))
# comparer = Comparer(ds, 0.7, algorithms, test_type=TestType.FULL_TRAINING)

# print(comparer)

# Scikit only

df = pd.read_excel('../datasets/acath.xls')
df = df.dropna()
X = df.drop(['sigdz', 'tvdlm'], axis=1)
y = df['sigdz']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)

algorithms = [
    SVC(kernel='linear'),
    SVC(kernel='sigmoid', coef0=0.5),
    SVC(kernel='sigmoid', coef0=0.8),
    SVC(kernel='sigmoid', coef0=1),
    SVC(kernel='rbf'),
    SVC(kernel='poly', degree=3),
    SVC(kernel='poly', degree=5),
]
for algorithm in algorithms:
    algorithm.fit(X_train, y_train)

    y_pred = algorithm.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

