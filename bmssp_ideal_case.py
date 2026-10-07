# bmssp_ideal_case.py
# Build a √n × √n grid (directed, constant degree ≤ 4), unit weights.
# Source is the top-left corner (0). We return (N0, start, edges).
# We deliberately SKIP transformGraph for this case, because the grid
# already has constant degree and we want to preserve equal-depth layers.

from __future__ import annotations
import math
from typing import List, Tuple

Edge = Tuple[int, int, int]  # (u, v, w)

def build_grid(n: int) -> Tuple[int, int, List[Edge]]:
    """
    Build a √n × √n grid (row-major ids), all edges weight 1, directed both ways
    horizontally and vertically (so degree at most 4 and still constant).

    If n is not a perfect square, we use L = floor(sqrt(n)) and keep only L*L nodes.

    Returns:
      N0:    number of vertices actually created (L*L)
      start: 0 (top-left corner)
      edges: list of (u, v, 1) directed edges
    """
    L = max(2, int(math.isqrt(max(1, n))))  # at least 2×2 to be interesting
    N0 = L * L
    start = 0
    edges: List[Edge] = []

    def vid(r: int, c: int) -> int:
        return r * L + c

    # Add unit-weight, constant-degree edges (undirected via two directed edges)
    for r in range(L):
        for c in range(L):
            u = vid(r, c)
            if c + 1 < L:  # right
                v = vid(r, c + 1)
                edges.append((u, v, 1))
                edges.append((v, u, 1))
            if r + 1 < L:  # down
                v = vid(r + 1, c)
                edges.append((u, v, 1))
                edges.append((v, u, 1))

    return N0, start, edges
