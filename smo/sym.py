import math
from config import *


class SYM:
    def __init__(self, s=None, n=None):
        self.txt = s or ""
        self.at = n or 0
        self.n = 0
        self.has = {}
        self.mode = None
        self.most = 0

    def add(self, x):
        if x == "?":
            return
        self.n += 1
        self.has[x] = 1 + self.has.get(x, 0)
        if self.has[x] > self.most:
            self.most, self.mode = self.has[x], x

    def mid(self):
        return self.mode

    def div(self):
        e = 0
        for v in self.has.values():
            e -= v / self.n * math.log(v / self.n, 2)
        return e

    def small(self):
        return 0

    def like(self, x, prior):
        return ((self.has.get(x, 0) or 0) + the["m"] * prior) / (self.n + the["m"])
