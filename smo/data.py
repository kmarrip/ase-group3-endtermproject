from cols import COLS
from rows import ROW
from utils import csv
import random
from sklearn import svm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def runHyper(c, e, x_train, x_test, y_train, y_test):
    c = float(c / 1000)
    e = float(e / 1000)
    regressor = svm.SVR(C=c, epsilon=e)
    regressor.fit(x_train, y_train)
    predicted = regressor.predict(x_test)
    return mean_absolute_error(y_test, predicted)


class DATA:
    def __init__(self, src, fun=None):
        self.rows = []
        self.cols = None
        self.mse_data = None
        if isinstance(src, str):
            csv(src, self.add)
        else:
            self.add(src, fun)

    def add(self, t, fun=None):
        row = t if isinstance(t, ROW) and t.cells else ROW(t)
        if self.cols:
            if fun:
                fun(row)
            self.rows.append(self.cols.add(row))
        else:
            self.cols = COLS(row)

    def mid(self, cols=None):
        u = {}
        for col in cols or self.cols.all:
            u[col.at] = col.mid()
        return ROW(u)

    def div(self, cols=None):
        u = {}
        for col in cols or self.cols.all:
            u[col.at] = col.div()
        return ROW(u)

    def stats(self, cols=None, fun=None, nDivs=None):
        u = {".N": len(self.rows)}
        for col in self.cols.y if cols is None else [self.cols.names[c] for c in cols]:
            cur_col = self.cols.all[col]
            u[cur_col.txt] = (
                round(getattr(cur_col, fun or "mid")(), nDivs)
                if nDivs
                else getattr(cur_col, fun or "mid")()
            )
        return u

    def shuffle(self, items):
        return random.sample(items, len(items))

    def getMSE(self, row):
        c = row.cells[0]
        e = row.cells[1]

        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.1
        )
        mse = runHyper(c, e, x_train, x_test, y_train, y_test)
        row.cells[2] = mse
        return mse

    def gate(self, budget0, budget, some, data1, x, y):
        self.mse_data = data1
        self.x = x
        self.y = y
        # self.mse_data = pd.read_csv(f"data/{file_name}_mse.csv")
        rows = self.shuffle(self.rows)
        print(len(rows))
        lite = rows[:budget0]
        dark = rows[budget0:]

        for _ in range(budget):
            lite.sort(key=lambda row: self.getMSE(row))
            n = int(len(lite) ** some)
            best, rest = lite[:n], lite[n:]
            todo = self.split(best, rest, lite, dark)
            lite.append(dark.pop(todo))
        return lite

    def split(self, best, rest, lite, dark):
        max_score = float("-inf")
        todo = 0

        bestDATA = DATA(self.cols.names)
        for row in best:
            bestDATA.add(row)

        restDATA = DATA(self.cols.names)
        for row in rest:
            restDATA.add(row)

        for i, row in enumerate(dark):
            b = row.like(bestDATA, len(lite), 2)
            r = row.like(restDATA, len(lite), 2)
            score = abs(b + r) / (abs(b - r) + 1e-30)
            if score > max_score:
                max_score, todo = score, i
        return todo
