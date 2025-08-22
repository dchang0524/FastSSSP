# dijkstra_baseline.py
import heapq
from typing import List, Dict

def dijkstra_on_adj_dict(adj: List[Dict[int, float]], start: int):
    """
    Standard Dijkstra on adjacency as list of dict: adj[u][v] = w (non-negative).
    Returns dist list (float).
    """
    n = len(adj)
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0.0
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

