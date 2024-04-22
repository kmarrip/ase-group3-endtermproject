# this is an abstract class for the num.py and sym.py
from sym import SYM
from num import NUM


class COLS:
    def __init__(self, txt: str):
        self.txt = txt.strip()
        if self.txt[0].isupper():
            self.col = NUM()
        else:
            self.col = SYM()
        self.heaven = 1 if self.txt.endswith("+") else 0

    def add(self, x):
        for val in x:
            self.col.add(val)

    def getNormalValues(self):
        return self.col.calculateNorm()
