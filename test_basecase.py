"""
test_basecase.py ─ BaseCase 단위 테스트
$ python test_basecase.py
"""

import math
import algorithms as alg   # 수정 중인 algorithms.py 모듈

# ────────────────────────────────────────────────────────────────
def build_tiny_graph():
    """
    0 ─5→ 1 ─10→ 2 ─15→ 3
    3 ─1→  1   (back-edge)
    """
    alg.N = 4
    alg.adj = [dict() for _ in range(alg.N)]
    edges = [(0, 1, 5), (1, 2, 10), (2, 3, 15), (3, 1, 1)]
    for u, v, w in edges:
        alg.adj[u][v] = w
    alg.M = len(edges)

    # sentinel = ∞  로 초기화
    INF = math.inf
    alg.dist  = [INF] * alg.N
    alg.depth = [INF] * alg.N
    alg.pred  = [-1]  * alg.N

    alg.start = 0
    alg.dist[alg.start]  = 0
    alg.depth[alg.start] = 0

    # k 값은 여유 있게
    alg.k = 3            # BaseCase 가 최대 4개까지 확정 가능
# ────────────────────────────────────────────────────────────────

def main():
    build_tiny_graph()

    B_init = math.inf
    S_init = {alg.start}

    B_prime, U = alg.BaseCase(B_init, S_init)

    print("BaseCase() 결과")
    print(f"  B'          = {B_prime}")
    print(f"  |U|          = {len(U)}")
    print("  U 정점 및 dist:")
    for v in sorted(U):
        print(f"    v={v:<2}   dist={alg.dist[v]}   pred={alg.pred[v]}")

if __name__ == "__main__":
    main()
