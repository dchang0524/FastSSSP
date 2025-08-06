# BMSSPTest.py
import algorithms as alg
import math

def small_graph():
    # 0→1→2→3 형태 + 몇 개 추가 간선
    alg.N = 4
    alg.adj = [dict() for _ in range(alg.N)]
    edges = [(0,1,5),(1,2,10),(2,3,15),(3,1,1)]
    for u,v,w in edges:
        alg.adj[u][v] = w
    alg.M = len(edges)
    alg.start = 0
    alg.dist  = [float('inf')]*alg.N
    alg.depth = [0]*alg.N
    alg.pred  = [-1]*alg.N
    alg.dist[0] = 0

    # k,t 재설정
    import math, inspect
    alg.k = math.floor(math.log2(alg.N) ** (1/3))
    alg.t = math.floor(math.log2(alg.N) ** (2/3))

def main():
    print("--- BMSSP unit test ---")
    small_graph()
    l = math.ceil(math.log2(alg.N)/alg.t) if alg.t else 0
    B_prime, U = alg.BMSSP(l, float('inf'), {alg.start})
    print(f"Returned |U|={len(U)}, B'={B_prime}")
    for v in sorted(U):
        print(f"v={v}, dist={alg.dist[v]}, pred={alg.pred[v]}")
    print("--- done ---")

if __name__ == "__main__":
    main()
