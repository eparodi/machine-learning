import os, sys; sys.path.append(os.getcwd())

from common.algorithms.svm import SVM
from common.experiment.experiment_comparer import TestType, Comparer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cls_images = [
    {
        "file": "../datasets/cielo.jpg",
        "class": "cielo",
    },
    {
        "file": "../datasets/vaca.jpg",
        "class": "vaca",
    },
    {
        "file": "../datasets/pasto.jpg",
        "class": "pasto",
    },
]

X_train_arr = []
y_train_arr = []
X_test_arr = []
y_test_arr = []

def image_to_df(path):
    colourImg = Image.open(path)
    width, height = colourImg.size
    # colourImg = colourImg.resize((int(width / 2), int(height / 2)), Image.ANTIALIAS)
    colourPixels = colourImg.convert("RGB")
    colourArray = np.array(colourPixels.getdata()).reshape(
        colourImg.size + (3,))
    allArray = colourArray.reshape((-1, 3))
    return pd.DataFrame(allArray, columns=["r", "g", "b"]), colourImg

def crop_images(kls, dataframe):
    df = dataframe.copy()
    df.loc[df['class'] != kls, ['r', 'g', 'b']] = 0
    return df.drop('class', axis=1)

print("Reading images")

for cls_image in cls_images:
    df, _ = image_to_df(cls_image["file"])
    X = df.copy()
    df["class"] = cls_image["class"]
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70)
    X_train_arr.append(X_train)
    y_train_arr.append(y_train)
    X_test_arr.append(X_test)
    y_test_arr.append(y_test)

X_train = pd.concat(X_train_arr)
X_test = pd.concat(X_test_arr)
y_train = pd.concat(y_train_arr)
y_test = pd.concat(y_test_arr)

print("SVM learning")
svm = SVC(kernel='linear', cache_size=1024*4, C=1.0)
# svm = SVC(kernel='sigmoid', coef0=0.5, cache_size=1024*4)
# svm = SVC(kernel='sigmoid', coef0=0.8, cache_size=1024*4)
# svm = SVC(kernel='sigmoid', coef0=1, cache_size=1024*4)
# svm = SVC(kernel='rbf', cache_size=1024*4, C=1.0, gamma=0.001)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Predicting image")
cow_df, image = image_to_df("../datasets/cow.jpg")
y_pred = svm.predict(cow_df)

print("Cropping images")
cow_df['class'] = y_pred
cielo_df = crop_images('cielo', cow_df)
vaca_df = crop_images('vaca', cow_df)
pasto_df = crop_images('pasto', cow_df)

print("Showing images")
width, height = image.size
Image.fromarray(np.uint8(cielo_df.values.reshape(height, width, 3))).save("sky.png")
Image.fromarray(np.uint8(vaca_df.values.reshape(
    height, width, 3))).save("cow.png")
Image.fromarray(np.uint8(pasto_df.values.reshape(
    height, width, 3))).save("grass.png")

print("Plot image distribution")
plot_df = df.copy()
plot_df['color'] = plot_df['class'].replace({
    "cielo": "blue",
    "vaca": "brown",
    "pasto": "green",
})
threedee = plt.figure().gca(projection='3d')
threedee.scatter(plot_df['r'], plot_df['g'],
                 plot_df['b'], c=plot_df['color'])
threedee.set_xlabel('r')
threedee.set_ylabel('g')
threedee.set_zlabel('b')
plt.show()
