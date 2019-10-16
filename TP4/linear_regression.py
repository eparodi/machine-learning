import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

def evaluate_by_rsq(X_test, X_train, y_train, y_test, rsq):
    change_rsq = True
    while change_rsq:
        change_rsq = False
        dropped_column = None
        print(len(X_test.columns))
        for col in X_test.columns:
            X_te = X_test.drop(col, axis=1)
            X_tr = X_train.drop(col, axis=1)
            lm_red = linear_model.LinearRegression()
            model = lm_red.fit(X_tr, y_train)
            predictions = lm_red.predict(X_te)
            new_rsq = lm_red.score(X_te, y_test)
            if new_rsq > rsq:
                rsq = new_rsq
                dropped_column = col
                change_rsq = True
        if dropped_column:
            X_test = X_test.drop(dropped_column, axis=1)
            X_train = X_train.drop(dropped_column, axis=1)
            print("Drop: ", dropped_column)
            print("Rsq:", rsq)


def evaluate_by_rss(X_test, X_train, y_train, y_test, rss):
    change_rss = True
    while change_rss:
        change_rss = False
        dropped_column = None
        print(len(X_test.columns))
        for col in X_test.columns:
            X_te = X_test.drop(col, axis=1)
            X_tr = X_train.drop(col, axis=1)
            lm_red = linear_model.LinearRegression()
            model = lm_red.fit(X_tr, y_train)
            predictions = lm_red.predict(X_te)
            new_rss = ((predictions - y_test) ** 2).sum()
            if new_rss < rss:
                rss = new_rss
                dropped_column = col
                change_rss = True
        if dropped_column:
            X_test = X_test.drop(dropped_column, axis=1)
            X_train = X_train.drop(dropped_column, axis=1)
            print("Drop: ", dropped_column)
            print("Rss:", rss)


df = pd.read_csv('data_multiple_regression_exercice.csv', sep=' ')
X = df.drop('weight', axis=1)
y = df['weight']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

rsq = lm.score(X_test, y_test)
rss = ((predictions - y_test) ** 2).sum()
print("Rsq:", rsq)
print("Rss:", rss)
evaluate_by_rsq(X_test, X_train, y_train, y_test, rsq)
evaluate_by_rss(X_test, X_train, y_train, y_test, rss)
