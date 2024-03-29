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
        "file": "../datasets/fur.png",
        "class": "cat",
    },
    {
        "file": "../datasets/grass.png",
        "class": "grass",
    },
]

classes = []

def image_to_df(path):
    colourImg = Image.open(path)
    width, height = colourImg.size
    colourImg = colourImg.resize((int(width / 2), int(height / 2)), Image.ANTIALIAS)
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
    df["class"] = cls_image["class"]
    classes.append(df)

df = pd.concat(classes)
df = df.drop_duplicates()

print("SVM learning")
svm = SVC(kernel='linear', cache_size=1024*4, C=1.0)
# svm = SVC(kernel='sigmoid', coef0=0.5, cache_size=1024*4)
# svm = SVC(kernel='sigmoid', coef0=0.8, cache_size=1024*4)
# svm = SVC(kernel='sigmoid', coef0=1, cache_size=1024*4)
# svm = SVC(kernel='rbf', cache_size=1024*4, C=1.0, gamma=0.1)
X_train = df.drop("class",axis=1)
y_train = df["class"]
svm.fit(X_train, y_train)

print("Predicting image")
cat_df, image = image_to_df("../datasets/cat.png")
y_pred = svm.predict(cat_df)

print("Cropping images")
cat_df['class'] = y_pred
catimg_df = crop_images('cat', cat_df)
grass_df = crop_images('grass', cat_df)

print("Showing images")
width, height = image.size
Image.fromarray(np.uint8(grass_df.values.reshape(height, width, 3))).save("grass2.png")
Image.fromarray(np.uint8(catimg_df.values.reshape(
    height, width, 3))).save("cat.png")

print("Plot image distribution")
plot_df = df.copy()
plot_df['color'] = plot_df['class'].replace({
    "cat": "gray",
    "grass": "green",
})
threedee = plt.figure().gca(projection='3d')
threedee.scatter(plot_df['r'], plot_df['g'],
                 plot_df['b'], c=plot_df['color'])
threedee.set_xlabel('r')
threedee.set_ylabel('g')
threedee.set_zlabel('b')
plt.show()
