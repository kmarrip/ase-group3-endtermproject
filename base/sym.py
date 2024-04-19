import math

class SYM:
    m = 0
    
    def __init__(self, s=None, n=None):
        self.txt = s or " "
        self.at = n or 0
        self.n = 0
        self.has = {} #this is a dictionary to keep track of the frequencies of the symbols
        self.mode = None #this is the symbol which repeats the most
        self.most = 0 #this is the frequency of the most repeating symbol
    
    def add(self, x):
        if x != "?":
            self.n += 1
            self.has[x] = 1+self.has.get(x,0)
            if self.has[x] > self.most:
                self.most, self.mode = self.has[x], x

    def mid(self): #mid for symbols is mode itself
        return self.mode
    
    def div(self):
        e = 0
        for v in self.has.values():
            e -= v / self.n * math.log(v / self.n, 2)
        return e

    def small(self):
        return 0

    def like(self, x, prior, the):
        return ((self.has.get(x, 0) + the.m * prior) / (self.n + the.m))
