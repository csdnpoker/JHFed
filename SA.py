import math
from random import random
import matplotlib.pyplot as plt



# x为公式里的x1,y为公式里面的x2
class SA:
    def __init__(self, func, maxa, maxb, T0=100, Tf=0.01, alpha=0.99):
        self.func = func
        self.iter = iter
        self.alpha = alpha
        self.T0 = T0
        self.Tf = Tf
        self.T = T0
        self.x = maxa
        self.y = maxb
        self.most_best = []
        self.history = {'f': [], 'T': []}
        self.lista = []
    def generate_new(self, x, y):  # 扰动产生新解的过程
        while True:
            x_new = x + self.T * (random() - random())
            y_new = y + self.T * (random() - random())
            if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
                break
        return x_new, y_new

    def Metrospolis(self, f, f_new):  # Metropolis准则
        if f_new <= f:
            return 1
        else:
            p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def best(self):
        f_list = []
        for i in range(self.iter):
            f = self.func(self.x[i], self.y[i])
            f_list.append(f)
        f_best = min(f_list)

        idx = f_list.index(f_best)
        return f_best, idx

    def run(self):
        count = 0

        while self.T > self.Tf:

            for i in range(self.iter):
                f = self.func(self.x[i], self.y[i])
                x_new, y_new = self.generate_new(self.x[i], self.y[i])
                f_new = self.func(x_new, y_new)
                if self.Metrospolis(f, f_new):
                    self.x[i] = x_new
                    self.y[i] = y_new
            ft, _ = self.best()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            self.T = self.T * self.alpha
            count += 1

        f_best, idx = self.best()

        self.lista.append(self.x[idx], self.y[idx])
        return self.lista

def func(x, y):  # 函数优化问题
    res = -(x*x+y*y)
    return res




