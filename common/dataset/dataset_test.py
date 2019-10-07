import random

import pandas as pd

import common.dataset.dataset_builder as db
import matplotlib.pyplot as plt


# ds = db.create_linearly_separable_dataset()
n = 20
height = 100
width = 100
center = 0.2

x1 = random.uniform(-(width*center)/2, (width*center)/2)
x2 = random.uniform(-(width*center)/2, (width*center)/2)
y1 = random.uniform(-(height*center)/2, (height*center)/2)
y2 = random.uniform(-(height*center)/2, (height*center)/2)

def point(x, y):
    return {"x": x, "y": y}

def pointInLine(x):
    return y1 + (y2 - y1) / (x2 - x1)*(x - x1)

def setLimits():
    plt.axis([-width / 2, width / 2, -height / 2, height / 2])

colors = ["r", "g"]
row_list = []
for x in range(0, n):
    data = {}
    data["x"] = random.uniform(-width/2, width/2)
    data["y"] = random.uniform(-height/2, height/2)
    d = (data["x"] - x1)*(y2 - y1) - (data["y"] - y1)*(x2 - x1)
    data["tag"] = 0 if d <= 0 else 1
    data["color"] = colors[data["tag"]]
    row_list.append(data)

summary = pd.DataFrame(row_list, columns=["tag", "x", "y", "color"])
line = pd.DataFrame([point(x1, y1), point(x2, y2),
                     point(-width/2, pointInLine(-width/2)),
                     point(width/2, pointInLine(width/2))], columns=["x", "y"])

ax = plt.gca()
setLimits()
line.plot(kind='line',x="x",y="y", ax=ax)

setLimits()
summary.plot(kind='scatter',x='x',y="y", color=summary["color"], ax=ax)

plt.show()

