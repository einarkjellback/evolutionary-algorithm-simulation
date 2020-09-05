from math import cos, sin, e, sqrt
from random import random
import matplotlib.pyplot as plt
from functools import partial


def random_individual():
    return [LOW + random() * (HIGH - LOW) for _ in range(DIMS)]


def ackley(x): return 3 * (cos(2 * x[0]) + sin(2 * x[1])) + e**(-0.2) * sqrt(x[0]**2 + x[1]**2)


def adaptiveHC(fitness_estimator: classmethod, mutation_chance=0.5):
    f = fitness_estimator
    x = random_individual()
    for gen in range(GENS):
        x_next = x.copy()
        for i in range(0, len(x)):
            if random() < mutation_chance:
                x_next[i] = LOW + random() * (HIGH - LOW)
        if f(x_next) > f(x):
            x = x_next
    return x


def steepest_ascentHC(fitness_estimator: classmethod):
    x = random_individual()
    for gen in range(GENS):
        x_nexts = [x.copy() for _ in range(DIMS)]
        for dim in range(DIMS):
            x_nexts[dim][dim] = LOW + random() * (HIGH - LOW)
        x_nexts.append(x)
        fitnesses = list(map(fitness_estimator, x_nexts))
        dim = fitnesses.index(max(fitnesses))
        x_next = x_nexts[dim]
        if x_next == x:
            x_next = random_individual()
        x = x_next
    return x


def next_ascentHC(fitness_estimator: classmethod):
    x = random_individual()
    for gen in range(GENS):
        replaceFlag = False
        x_next = x.copy()
        for dim in range(DIMS):
            x_next[dim] = LOW + random() * (HIGH - LOW)
            if fitness_estimator(x_next) > fitness_estimator(x):
                x = x_next
                replaceFlag = True
        if not replaceFlag:
            x = random_individual()
    return x

def random_mutationHC(fitness_estimator: classmethod):
    x = random_individual()
    for gen in range(GENS):
        x_next = x.copy()
        dim = int(random() * DIMS)
        x_next[dim] = LOW + random() * (HIGH - LOW)
        if fitness_estimator(x_next) > fitness_estimator(x):
            x = x_next
    return x


def run_simulations(simulations: int, fitness_estimator: classmethod, HC_algorithm: classmethod):
    mean = 0
    for sim in range(simulations):
        x = HC_algorithm(fitness_estimator)
        mean += fitness_estimator(x)
    mean /= simulations
    return mean


def fitness_estimator(v): return -ackley(v)


GENS = 1000
SIMS = 20
RESOLUTION = 10
ps = [i / RESOLUTION for i in range(1, RESOLUTION + 1)]
LOW, HIGH = -30, 30
DIMS = 2
minimum_values = []
HC_algs = ['adaptiveHC', 'steepest_ascentHC', 'next_ascentHC', 'random_mutationHC']
for p in ps:
    partial_adaptive = partial(adaptiveHC, mutation_chance=p)
    minimum_values.append(-run_simulations(SIMS, fitness_estimator, partial_adaptive))
partial_run_sim = partial(run_simulations, SIMS, fitness_estimator)
results_HC = list(map(lambda f: partial_run_sim(eval(f)), HC_algs[1:]))
results_HC.insert(0, min(minimum_values))

plt.figure()
plt.bar(list(map(str, HC_algs)), results_HC)
plt.show()




