import ast
from pathlib import Path


def coerce(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return x.strip()


class DATA:
    def __init__(self, src):
        self.rows, self.cols = [], None
        if isinstance(src, str) == False:
            raise Exception("Data source should be a string")

        fileDescriptor = Path(src)
        if fileDescriptor.exists() == False:
            raise FileNotFoundError()
        if fileDescriptor.suffix != ".csv":
            raise Exception("data file should be csv")
        firstLine = False
        with open(fileDescriptor.absolute(), "r", encoding="utf-8") as file:
            for line in file:
                row = list(map(coerce, line.strip().split(",")))
                print(f"row value is {row}")
                break
                self.add(row)

    def add(self, r):
        row = ROW(r)
        # row =r if 'cells' in r else ROW(r)
        if self.cols:
            if fun:
                fun(self, row)
            self.rows.append(self.cols.add(row))
        else:
            self.cols = COLS(row)

    def mid(self, cols=None):
        u = [col.mid() for col in (cols or self.cols.all)]
        return ROW(u)

    def div(self, cols=None):
        u = [col.div() for col in (cols or self.cols.all)]
        return ROW(u)

    def small(self):
        u = [col.small() for col in self.cols.all]
        return ROW(u)

    def stats(self, cols=None, fun=None, ndivs=None):
        u = {".N": len(self.rows)}
        for col in self.cols.y if cols is None else [self.cols.names[c] for c in cols]:
            current_col = self.cols.all[col]
            u[current_col.txt] = (
                round(getattr(current_col, fun or "mid")(), ndivs)
                if ndivs
                else getattr(current_col, fun or "mid")()
            )
        return u


data = DATA("../data/wine.csv")
