from math import sqrt
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from cols import COLS
import matplotlib.pyplot as plt
import time
from os import system, name

def clear():
 
    # for windows
    if name == 'nt':
        _ = system('cls')
 
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

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
    best_c = ''
    best_e = ''
    least_mse = 1e10
    start = time.time()
    for c in range(1, 100):
        for e in range(1, 100):
            some_start = time.time()
            print(f"hyper params are {c} and {e}")
            currentMSE = runHyper(c,e,x_train,x_test,y_train,y_test)
            if currentMSE < least_mse:
                best_c = c
                best_e = e
                least_mse = currentMSE

            accuracyDF.loc[len(accuracyDF.index)] = [
                c/1000,
                e/1000,
                runHyper(c, e, x_train, x_test, y_train, y_test),
            ]
            some_end = time.time()
            print(f" iteration took {some_end - some_start}")
    
    clear() 
    print(f"best MSE is {least_mse}, for c={best_c}, e={best_e}")

    end = time.time()
    print(f"{filePath} This took {end - start} time ")

    print(accuracyDF.head(100))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(accuracyDF["c"], accuracyDF["e"], accuracyDF["error"])
    ax.set_xlabel('c')
    ax.set_ylabel('e')
    ax.set_zlabel("MSE")
    ax.set_title(filePath)

    plt.show()


# grid search
run('../data/SS-C.csv',3)
run('../data/SS-H.csv',4)
run('../data/auto93.csv',5)
run('../data/pom3a.csv',9)
run('../data/wine.csv',10)
run('../data/SS-A.csv',3)
run('../data/dtlz2.csv',10)
run('../data/dtlz3.csv',10)
run('../data/dtlz4.csv',10)
run('../data/dtlz5.csv',10)
run('../data/dtlz6.csv',10)
run('../data/dtlz7.csv',10)

# random search
space = dict()
space["c"] = list(map(lambda x: x / 100, range(100)))
space["epsilon"] = list(map(lambda x: x / 100, range(100)))
