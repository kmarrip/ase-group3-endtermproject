import optuna
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn.metrics import mean_absolute_error
import math
from cols import COLS
import time
from os import system, name

def clear():
 
    # for windows
    if name == 'nt':
        _ = system('cls')
 
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')


def runHyper(c, e, x_train, x_test, y_train, y_test):
    regressor = svm.SVR(C=c, epsilon=e)
    regressor.fit(x_train, y_train)
    predicted = regressor.predict(x_test)
    return mean_absolute_error(y_test, predicted)


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
    return list(map(lambda x: math.sqrt(x / nmCols), d2h))


def runOptuna(n:int, path:str, iters = 1000):
    #optuna.logging.set_verbosity(optuna.logging.ERROR)
    def objective(trial):
        df = pd.read_csv(path)
        x = df.iloc[:, 0:n].values
        y = df.iloc[:, n:]

        y = getDistance2HeavenArray(y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        c = trial.suggest_float("c", 0, 0.1)
        e = trial.suggest_float("e", 0, 0.1)

        meanError = runHyper(c, e, x_train, x_test, y_train, y_test)
        return meanError


    study = optuna.create_study(direction="minimize")
    start = time.time()
    study.optimize(objective, n_trials=iters)
    end = time.time() 

    
    trial = study.best_trial

    clear()
    print(f"time took is {end - start}")
    print(f"For the dataset {path},best mean squared error is {trial.value}, best hyperparmas {trial.params}")
    delee = input('enter to continue')


runOptuna(3,'../data/SS-C.csv')
runOptuna(4,'../data/SS-H.csv')
runOptuna(5,'../data/auto93.csv')
runOptuna(9,'../data/pom3a.csv')
runOptuna(10,'../data/wine.csv')
runOptuna(3,'../data/SS-A.csv')
runOptuna(10,'../data/dtlz2.csv')
runOptuna(10,'../data/dtlz3.csv')
runOptuna(10,'../data/dtlz4.csv')
runOptuna(10,'../data/dtlz5.csv')
runOptuna(10,'../data/dtlz6.csv')
runOptuna(10,'../data/dtlz7.csv')
