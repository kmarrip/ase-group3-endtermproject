from utils import egs, the
from data import DATA
import math
import math
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from config import the
import time


def main():
    for action, _ in egs.items():
        if the["todo"] == "all" or the["todo"] == action:
            for key, value in saved_options.items():
                the[key] = value

            global Seed
            Seed = the["seed"]


def smoFile(n, fileName):
    start = time.time()
    main()
    paramsX = pd.DataFrame()
    paramsX["c"] = ""
    paramsX["e"] = ""
    cValue = []
    eValue = []
    for c in range(1, 101):
        for e in range(1, 101):
            cValue.append(c)
            eValue.append(e)
    paramsX["c"] = cValue
    paramsX["e"] = eValue

    paramsX["mse"] = ""
    paramsX.to_csv("params.csv", index=False)
    paramsX = DATA("./params.csv")
    os.remove("params.csv")
    data1 = pd.read_csv(fileName)
    # data1 = DATA
    print(data1.shape)

    x = data1.iloc[:, :n]
    y = data1.iloc[:, n:]

    for col in data1.columns:
        if col.endswith("+"):
            data1[col] = (data1[col] - data1[col].min()) / (
                data1[col].max() - data1[col].min()
            )
            data1[col] = 1 - data1[col]
        if col.endswith("-"):
            data1[col] = (data1[col] - data1[col].min()) / (
                data1[col].max() - data1[col].min()
            )
            data1[col] = 0 - data1[col]
    columns_to_drop = [
        col for col in data1.columns if col.endswith("+") or col.endswith("-")
    ]

    n = len(columns_to_drop)

    calDistance = lambda row: math.sqrt(
        sum(
            (
                row[col] ** 2
                for col in data1.columns
                if col.endswith("+") or col.endswith("-")
            )
        )
        / n
    )
    data1["d2h"] = data1.apply(calDistance, axis=1)

    data1 = data1.drop(columns=columns_to_drop)
    iters = 1
    the["k"] = 1
    the["m"] = 2
    for _ in range(iters):
        start = time.time()
        lite = paramsX.gate(50, 20, 0.5, data1, x, data1["d2h"])
        X, y = [], []
        lite.pop()
        for row in lite:
            c, e, mse = row.cells[0], row.cells[1], row.cells[2]
            X.append([c, e])
            y.append(mse)
        lr = LinearRegression()
        lr.fit(X, y)

        param = pd.DataFrame()
        param["c"] = list(range(1, 100))
        param["e"] = list(range(1, 100))

        param["mse"] = lr.predict(param)
        print(param[param["mse"] == param["mse"].min()])

    end = time.time()
    print(f"{fileName} took {end - start}")


smoFile(3, "../data/SS-C.csv")
smoFile(4, "../data/SS-H.csv")
smoFile(5, "../data/auto93.csv")
smoFile(9, "../data/pom3a.csv")
smoFile(10, "../data/wine.csv")
smoFile(3, "../data/SS-A.csv")
smoFile(10, "../data/dtlz2.csv")
smoFile(10, "../data/dtlz3.csv")
smoFile(10, "../data/dtlz4.csv")
smoFile(10, "../data/dtlz5.csv")
smoFile(10, "../data/dtlz6.csv")
smoFile(10, "../data/dtlz7.csv")
