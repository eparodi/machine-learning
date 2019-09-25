from common.algorithms.svm import SVM
from common.experiment.experiment_comparer import TestType, Comparer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

import pandas as pd
import numpy as np

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

classes = []

def image_to_df(path):
    colourImg = Image.open(path)
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
svm = SVC(kernel='linear', cache_size=1024*4)
X_train = df.drop("class",axis=1)
y_train = df["class"]
svm.fit(X_train, y_train)

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