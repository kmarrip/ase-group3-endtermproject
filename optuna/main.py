import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import svm
import time
from pathlib import Path
from cols import COLS
import math
import os


def clear():
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")


def runHyper(c, e, x_train, x_test, y_train, y_test):
    regressor = svm.SVR(C=c, epsilon=e)
    regressor.fit(x_train, y_train)
    predicted = regressor.predict(x_test)
    return mean_absolute_error(y_test, predicted)


def getDistance2HeavenArray(y):
    columns = y.columns.values
    cols = []
    for colName in columns:
        col = COLS(colName)
        col.add(list(y[colName]))
        cols.append(col)
    for col in cols:
        col.values = col.getNormalValues()
        squared = [(col.heaven - x) ** 2 for x in col.values]
        col.values = squared
    d2h = [
        sum([cols[c].values[i] for c in range(len(cols))])
        for i in range(len(cols[0].values))
    ]
    return list(map(lambda x: math.sqrt(x / len(cols)), d2h))


def runOptuna(n: int, path: str, output_dir=Path("optuna_results"), iters=100):
    def objective(trial):
        df = pd.read_csv(path)
        x = df.iloc[:, 0:n].values
        y = df.iloc[:, n:]
        y = getDistance2HeavenArray(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        c = trial.suggest_float("c", 0.0001, 0.1)
        e = trial.suggest_float("e", 0.0001, 0.1)
        return runHyper(c, e, x_train, x_test, y_train, y_test)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=iters)

    # Collect and save results
    results = {
        "Trial": [trial.number for trial in study.trials],
        "MSE": [trial.value for trial in study.trials],
        "C": [trial.params["c"] for trial in study.trials],
        "Epsilon": [trial.params["e"] for trial in study.trials],
    }
    results_df = pd.DataFrame(results)
    output_dir.mkdir(exist_ok=True, parents=True)
    results_file = output_dir / f"{Path(path).stem}_optuna_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    best_trial = study.best_trial
    print(f"Best MSE: {best_trial.value} with params {best_trial.params}")


runOptuna(3, "../data/SS-C.csv")
runOptuna(4, "../data/SS-H.csv")
runOptuna(5, "../data/auto93.csv")
runOptuna(9, "../data/pom3a.csv")
runOptuna(10, "../data/wine.csv")
runOptuna(3, "../data/SS-A.csv")
runOptuna(10, "../data/dtlz2.csv")
runOptuna(10, "../data/dtlz3.csv")
runOptuna(10, "../data/dtlz4.csv")
runOptuna(10, "../data/dtlz5.csv")
runOptuna(10, "../data/dtlz6.csv")
runOptuna(10, "../data/dtlz7.csv")
# Example usage for a single dataset
