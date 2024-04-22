from num import NUM
from sym import SYM


class COLS:
    def __init__(self, row):
        self.x, self.y, self.all = {}, {}, []
        self.klass = None
        for at, txt in enumerate(row.cells):
            if txt[0].isupper():
                col = NUM(txt, at)
            else:
                col = SYM(txt, at)

            self.all.append(col)

            if txt.endswith("X"):
                continue

            if txt.endswith("!"):
                self.klass = col
            if txt.endswith(("!", "+", "-")):
                self.y[at] = col
            else:
                self.x[at] = col
        self.names = row.cells

    def add(self, row):
        for cols in (self.x, self.y):
            for col in cols.values():
                col.add(row.cells[col.at])
        return row
