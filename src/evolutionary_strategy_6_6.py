from math import sqrt
from typing import List
import numpy as np
import matplotlib.pyplot as plt

std = 0.1 / (2 * sqrt(3))
variance = std * std
SIMS = 10  # 50
c = [0.6, 0.8, 1.0]
domain = lo, hi = -5.12, 5.12
DIMS = 10
GENS = 100  # 500
G = 10


def cost(x) -> float:
    return sum(map(lambda v: v ** 2, x))


avg_costs = [0 for _ in range(GENS)]


def run_es() -> List[float]:
    x = np.random.normal(0, variance, DIMS)
    success_history: bin = 0b0
    cost_history = [cost(x)]
    for gen in range(0, GENS):
        x_child = np.add(np.random.normal(0, variance, DIMS))
        x_child = map(lambda v: min(hi, v), x_child)
        x_child = list(map(lambda v: max(lo, v), x_child))
        success_history = success_history << 1
        if cost(x_child) < cost(x):
            x = x_child
            success_history += 0b1
        success_proportion = len([b for b in success_history[:G:-1] if b == '1']) / G
        if success_proportion < 0.20:
            pass
            #Std =

for sim in range(0, SIMS):
    avg_costs = np.add(avg_costs, run_es())
avg_costs = np.divide(avg_costs, SIMS)
