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

    # --- 변형 여부 스위치 ---------------------------------------
    USE_TRANSFORM = False   # ← transformGraph() 잠깐 끄고 테스트할 때 False

    if USE_TRANSFORM:
        alg.transformGraph()
        # 변형 후 배열 재초기화
        INF = math.inf
        alg.dist   = [INF] * alg.N
        alg.depth  = [INF] * alg.N
        alg.pred   = [-1]  * alg.N
        alg.dist[alg.start]  = 0
        alg.depth[alg.start] = 0
    else:
        # ★ 변형을 생략하면 adj 형식을 dict로 강제 변환
        #    list[set((v,w))]  ->  list[dict{v:w}]
        if alg.adj and isinstance(alg.adj[0], set):
            alg.adj = [{v: w for (v, w) in nbrs} for nbrs in alg.adj]
        # N, start, dist/depth/pred 는 build_graph_from_edges()에서 이미 초기화됨

    # 테스트용: 한 번에 모두 확정하도록 k = N'
    alg.k = math.floor(math.log2(alg.N) ** (1/3))
    alg.t = math.floor(math.log2(alg.N) ** (2/3))
    l     = math.ceil(math.log2(alg.N) / alg.t)

    B_prime, U = alg.BMSSP(l, math.inf, {alg.start})
    print(f"B'={B_prime}, |U|={len(U)}")
    for v in list(U)[:10]:
        print(v, alg.dist[v], alg.depth[v], alg.pred[v])
    print(alg.dist[:100])
    print(alg.k, alg.t, l)    

if __name__ == "__main__":
    main()