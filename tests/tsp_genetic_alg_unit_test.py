import unittest

from tsp_genetic_alg import generate_graph_degree


def dfs_count(node, graph, visited) -> int:
    visited[node] = True
    acc = 1
    for adj in graph[node]:
        if not visited[adj]:
            acc += dfs_count(adj, graph, visited)
    return acc


class MyTestCase(unittest.TestCase):
    def test_generate_graph_degree_is_connected(self):
        for size in range(2, 10):
            for degree in range(1, size + 1):
                for test in range(0, 50):
                    with self.subTest((size, degree, test)):
                        graph = generate_graph_degree(size, max_degree=degree)
                        visited_nodes = dfs_count(0, graph, [False for _ in range(size)])
                        self.assertEqual(size, visited_nodes)


if __name__ == '__main__':
    unittest.main()
