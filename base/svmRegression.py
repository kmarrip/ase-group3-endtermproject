from math import sqrt
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from cols import COLS
import matplotlib.pyplot as plt
import time

def getDistance2HeavenArray(y):
    columns = y.columns.values
    cols = []
    for colName in columns:
        cols.append(COLS(colName))
    for i in range(len(cols)):
        cols[i].add(list(y.iloc[:, i]))
    for col in cols:
        col.values = col.getNormalValues()

    for col in cols:
        squared = []
        for x in col.values:
            squared.append((col.heaven - x) ** 2)
        col.values = squared
    d2h = []
    n = len(cols[0].values)
    nmCols = len(cols)
    for i in range(n):
        temp = 0
        for c in range(len(cols)):
            temp += cols[c].values[i]
        d2h.append(temp)
    return list(map(lambda x: sqrt(x / nmCols), d2h))


def runHyper(c, e, x_train, x_test, y_train, y_test):
    c = float(c/1000)
    e = float(e/1000)
    regressor = svm.SVR(C=c, epsilon=e)
    regressor.fit(x_train, y_train)
    predicted = regressor.predict(x_test)
    return mean_absolute_error(y_test, predicted)


def run(filePath, n: int):
    df = pd.read_csv(filePath)
    x = df.iloc[:, 0:n].values  # the first n are independent variables
    y = df.iloc[:, n:]  # the last 2 are depended variables
    # we need to calculate the distance to heaven for each of the rows
    # and add to

    y = getDistance2HeavenArray(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    accuracyDF = pd.DataFrame()
    accuracyDF["c"] = ""
    accuracyDF["e"] = ""
    accuracyDF["error"] = ""
    
    start = time.time()
    for c in range(1, 100):
        for e in range(1, 100):
            some_start = time.time()
            print(f"hyper params are {c} and {e}")
            accuracyDF.loc[len(accuracyDF.index)] = [
                c/1000,
                e/1000,
                runHyper(c, e, x_train, x_test, y_train, y_test),
            ]
            some_end = time.time()
            print(f" iteration took {some_end - some_start}")
    
    end = time.time()
    print(f"This took {end - start} time ")
    print(accuracyDF.head(100))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(accuracyDF["c"], accuracyDF["e"], accuracyDF["error"])
    plt.show()


# grid search
run('../data/wine.csv',10)
run('../data/SS-A.csv',3)
run('../data/dtlz2.csv',10)
run("../data/xomo_flight.csv",27)
# random search
space = dict()
space["c"] = list(map(lambda x: x / 100, range(100)))
space["epsilon"] = list(map(lambda x: x / 100, range(100)))
