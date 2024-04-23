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
    if name == "nt":
        _ = system("cls")

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system("clear")


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
    c = float(c / 1000)
    e = float(e / 1000)
    regressor = svm.SVR(C=c, epsilon=e)
    regressor.fit(x_train, y_train)
    predicted = regressor.predict(x_test)
    return mean_absolute_error(y_test, predicted)


from pathlib import Path

def run(filePath, n: int):
    df = pd.read_csv(filePath)
    x = df.iloc[:, 0:n].values
    y = df.iloc[:, n:]
    y = getDistance2HeavenArray(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    accuracyDF = pd.DataFrame(columns=["c", "e", "mse"])
    best_c, best_e, least_mse = "", "", float('inf')

    for c in range(1, 100):
        for e in range(1, 100):
            mse = runHyper(c, e, x_train, x_test, y_train, y_test)
            new_row = pd.DataFrame({"c": [c/1000], "e": [e/1000], "mse": [mse]})
            accuracyDF = pd.concat([accuracyDF, new_row], ignore_index=True)
            if mse < least_mse:
                best_c, best_e, least_mse = c, e, mse

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{Path(filePath).stem}_results.csv"
    accuracyDF.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


    # Plot the results
    plot_results(accuracyDF, filePath)

def plot_results(df, filePath):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(df["c"], df["e"], df["mse"])
    ax.set_xlabel("C values")
    ax.set_ylabel("Epsilon values")
    ax.set_zlabel("MSE")
    ax.set_title(filePath)
    plt.show()

# Example usage
run(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\data\auto93.csv", 5)
run(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\data\SS-C.csv", 3)
run(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\data\SS-H.csv", 4)
run(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\data\pom3a.csv", 9)
run(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\data\wine.csv", 10)
run(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\data\SS-A.csv", 3)
run(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\data\dtlz2.csv", 10)
run(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\data\dtlz3.csv", 10)
run(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\data\dtlz4.csv", 10)
run(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\data\dtlz5.csv", 10)
run(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\data\dtlz6.csv", 10)
run(r"C:\Users\saivi\OneDrive\Desktop\ase-group3-endtermproject\data\dtlz7.csv", 10)

