from typing import List, Set
import numpy as np


def generate_graph_passes(size: int, passes: int = 1) -> List[Set[int]]:
    graph = [set() for _ in range(size)]
    for _ in range(passes):
        unvisited_nodes = [i for i in range(size)]
        from_: int = np.random.choice(unvisited_nodes)
        while len(unvisited_nodes) > 1:
            unvisited_nodes.remove(from_)
            to: int = np.random.choice(unvisited_nodes)
            graph[from_].add(to)
            graph[to].add(from_)
            from_ = to
    return graph


def generate_graph_degree(size: int, max_degree: int = 1) -> List[Set[int]]:
    if max_degree > size: ValueError("max_degree must be less or equal to size")
    graph = [set() for _ in range(size)]
    graph[0].add(np.random.choice(range(1, size)))
    for from_ in range(size - 1, -1, -1):
        degree = np.random.choice(range(1, max_degree + 1))
        degree = max(0, degree - len(graph[from_]))
        degree = min(degree, from_)
        to_list = np.random.choice(range(from_), size=degree, replace=False)
        for to in to_list:
            if len(graph[to]) < max_degree:
                graph[from_].add(to)
                graph[to].add(from_)
    return graph


print(generate_graph_degree(5, 2))
