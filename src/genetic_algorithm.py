from random import random
import matplotlib.pyplot as plt
import numpy as np


def sphere(x: list) -> float: return sum(list(map(lambda v: v ** 2, x)))


def fitness(x: list) -> float: return 1 / sphere(x)


def random_sample(): return [lo + random() * (hi - lo) for _ in range(dims)]


def choose_parent(parents):
    fitnesses = list(map(fitness, parents))
    r = random() * sum(fitnesses)
    i = 0
    while r > 0:
        r -= fitnesses[i]
        i += 1
    return parents[i - 1]


def run_ga():
    min_costs, avg_costs = [], []
    parents = [random_sample() for _ in range(pop_size)]
    for gen in range(gens):
        costs = list(map(sphere, parents))
        min_costs.append(min(costs))
        avg_costs.append(sum(costs) / len(costs))
        children = []
        while len(children) < len(parents):
            parent_a = choose_parent(parents)
            parent_b = choose_parent(parents)
            crossover = int(random() * (dims + 1))
            child_a = parent_a[:crossover]
            child_a.extend(parent_b[crossover:])
            child_b = parent_b[:crossover]
            child_b.extend(parent_a[crossover:])
            children.extend([child_a, child_b])
        for i in range(len(children)):
            child = children[i]
            for dim in range(dims):
                if random() < mutation_rate:
                    child[dim] = lo + random() * (hi - lo)
            children[i] = child
        parents = children
    return min_costs, avg_costs


domain = lo, hi = [-5, 5]
dims = 20
gens = 200
pop_size = 40
mutation_rate = 0.01
sims = 5

"""
min_costs, avg_costs = run_ga()
plt.figure()
plt.title('Best and average costs for 1 run of genetic algorithm')
plt.xlabel('Generation')
plt.ylabel('Cost')
plt.plot(range(gens), min_costs, label='Min costs')
plt.plot(range(gens), avg_costs, label='Avg costs')
plt.legend()
"""

min_costs_acc, avg_costs_acc = run_ga()
for sim in range(sims - 1):
    min_costs, avg_costs = run_ga()
    min_costs_acc = np.add(avg_costs_acc, min_costs)
    avg_costs_acc = np.add(avg_costs_acc, avg_costs)
min_costs = np.divide(min_costs_acc, sims)
avg_costs = np.divide(avg_costs_acc, sims)

plt.figure()
plt.title('Best and average costs for n runs of genetic algorithm')
plt.xlabel('Generation')
plt.ylabel('Cost')
plt.plot(range(gens), min_costs, label='Min costs')
plt.plot(range(gens), avg_costs, label='Avg costs')
plt.legend()
plt.grid(axis='y', linestyle="dashed")
plt.show()