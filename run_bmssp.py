import math
import algorithms as alg
import testcase_generator as tg

def build_graph_from_edges(edge_list, n):
    # --- set((v,w)) 형식으로 adj 생성 --------------------------
    alg.N   = n
    alg.adj = [set() for _ in range(n)]
    for u, v, w in edge_list:
        alg.adj[u].add((v, w))
    alg.M = len(edge_list)

    # 임시(변형 전) SSSP 배열
    INF = math.inf
    alg.dist  = [INF] * n
    alg.depth = [INF] * n
    alg.pred  = [-1]  * n
    alg.start = 0
    alg.dist[alg.start]  = 0
    alg.depth[alg.start] = 0

def main():
    edges = tg.adj
    N     = max(max(u, v) for u, v, _ in edges) + 1
    build_graph_from_edges(edges, N)

    # --- 필수: 그래프 변형 -------------------------------------
    alg.transformGraph()

    # --- 변형 후 배열 재초기화 ---------------------------------
    INF = math.inf
    alg.dist   = [INF] * alg.N
    alg.depth  = [INF] * alg.N
    alg.pred   = [-1]  * alg.N
    alg.dist[alg.start]  = 0
    alg.depth[alg.start] = 0

    # 테스트용: 한 번에 모두 확정하도록 k = N'
    alg.k = math.floor(math.log2(alg.N) ** (1/3))
    alg.t = math.floor(math.log2(alg.N) ** (2/3))
    l     = math.ceil(math.log2(alg.N) / alg.t)

    B_prime, U = alg.BMSSP(l, math.inf, {alg.start})
    print(f"B'={B_prime}, |U|={len(U)}")
    for v in list(U)[:10]:
        print(v, alg.dist[v])

if __name__ == "__main__":
    main()

