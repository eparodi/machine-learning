import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('../datasets/data_multiple_regression_exercice.csv', sep=' ')
X = df.drop('weight', axis=1)
y = df['weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
test_arr = y_test.array
for i in range(len(predictions)):
    print("{p} - {v}".format(p=predictions[i], v=test_arr[i]))
print("Rsq:", lm.score(X_test, y_test))
