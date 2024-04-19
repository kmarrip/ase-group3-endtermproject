class NUM:
    def __init__(self, s=None, n=None):
        self.n = 0
        self.mu = 0
        self.m2 = 0
        self.hi = -1e30
        self.lo = 1e30
        self.values = []

    def add(self, x):
        if x != "?":
            self.n += 1
            self.lo = min(x, self.lo)
            self.hi = max(x, self.hi)
            self.values.append(x)

    def calculateNorm(self):
        return list(map(lambda x: (x - self.lo) / (self.hi - self.lo), self.values))

    def mid(self):
        return self.mu

    def div(self):
        return 0 if self.n < 2 else (self.m2 / (self.n - 1)) ** 0.5

    def small(self, the):
        return the.cohen * self.div()

    def norm(self, x):
        return x if x == "?" else (x - self.lo) / (self.hi - self.lo + 1e-30)

    def like(self, x):
        mu, sd = self.mid(), self.div() + 1e-30
        nom = 2.718 ** (-0.5 * (x - mu) ** 2 / (sd**2))
        denom = sd * 2.5 + 1e-30
        return nom / denom
