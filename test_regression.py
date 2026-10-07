"""Small correctness checks for the BMSSP implementation and its hot path."""

import math
import random
import unittest

import algorithms as alg
from D import DataStructureD
from dijkstra_baseline import dijkstra_on_adj_dict


class FastSSSPRegressionTests(unittest.TestCase):
    def test_singleton_blocks_and_pull(self):
        bound = (math.inf, math.inf, math.inf)
        queue = DataStructureD(1, bound)
        queue.insert(1, (5, 0, 1))
        queue.insert(2, (2, 0, 2))
        queue.insert(3, (4, 0, 3))
        self.assertEqual(queue.pull(), ((4, 0, 3), {2}))
        queue.batch_prepend([(4, (1, 0, 4))])
        self.assertEqual(queue.pull(), ((4, 0, 3), {4}))
        self.assertEqual(queue.pull(), ((5, 0, 1), {3}))
        self.assertEqual(queue.pull(), (bound, {1}))
        self.assertEqual(queue.pull(), (bound, set()))

    def test_bmssp_matches_dijkstra(self):
        rng = random.Random(42)
        for n in (2, 5, 10, 30):
            for _ in range(20):
                graph = [{} for _ in range(n)]
                for u in range(n):
                    for v in range(n):
                        if u != v and rng.random() < min(0.3, 4 / n):
                            graph[u][v] = rng.randrange(10)

                alg.N = n
                alg.adj = graph
                alg.start = 0
                alg.k = max(1, math.floor(math.log2(n) ** (1 / 3)))
                alg.t = max(1, math.floor(math.log2(n) ** (2 / 3)))
                alg.dist = [math.inf] * n
                alg.depth = [math.inf] * n
                alg.pred = [-1] * n
                alg.dist[0] = alg.depth[0] = 0

                level = math.ceil(math.log2(n) / alg.t)
                _, complete = alg.BMSSP(level, (math.inf,) * 3, {0})
                expected = dijkstra_on_adj_dict(graph, 0)
                self.assertEqual(alg.dist, expected)
                self.assertEqual(complete, {v for v, d in enumerate(expected) if d < math.inf})


if __name__ == "__main__":
    unittest.main()
