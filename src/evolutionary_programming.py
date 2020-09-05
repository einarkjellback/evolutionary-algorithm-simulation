from random import random
from typing import Callable, Tuple
import matplotlib.pyplot as plt


def cost(x: list) -> float:
    return sum(map(lambda i: i ** 2, x))


def cost_norm(x: list) -> float:
    c = cost(x)
    return 1 + (c / (c + 1)) ** 10


def var_a(x: list) -> float: return (beta * cost_norm(x)) ** (1 / 2)


def var_b(x: list) -> float: return beta * (cost_norm(x) ** 2)


def var_c(x: list) -> float: return beta * (cost_norm(x) ** (1 / 2))


def mutate(x: list, var: Callable) -> list:
    c = var(x)
    x_new = []
    for i in range(len(x)):
        x_i = x[i]
        x_new.append(x_i + (lo + random() * (hi - lo) * c))
    return x_new


def run_ep(var: Callable):
    pop = []
    best_xs = []
    avg_xs = []
    for _ in range(pop_size):
        pop.append([lo + random() * (hi - lo) for _ in range(dims)])
    for _ in range(gens):
        pop.extend(list(map(lambda x: mutate(x, var), pop)))
        pop = sorted(pop, key=cost)
        pop = pop[:gens]
        best_xs.append(cost(pop[0]))
        #avg_xs.append(sum(map(cost, pop)) / len(pop))
    return best_xs#, avg_xs


domain = lo, hi = -5.12, 5.12
beta = 1.024
pop_size = 50
gens = 50
sims = 20
dims = 10

lowest_costs = []
for _ in range(sims):
    lowest_costs.append(run_ep(var_a)[0])
best = sum(lowest_costs) / sims
print(best)

