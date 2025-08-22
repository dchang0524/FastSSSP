# benchmark_compare.py
import os
import re
import math
import time
import gc
import importlib
import types
import random
from typing import List, Tuple
import matplotlib.pyplot as plt

# 로컬 모듈
import algorithms as alg            # 당신의 BMSSP/변환 코드
from dijkstra_baseline import dijkstra_on_adj_dict

# ------------------------------------------------------------
# 1) testcase_generator.py를 "그대로" 사용하되, N만 바꿔서 실행
#    - run_bmssp와 동일한 생성기를 쓰기 위해 소스 읽어 exec
# ------------------------------------------------------------
def load_edges_with_generator(N: int, seed: int = 12345) -> Tuple[int, List[Tuple[int,int,int]]]:
    """
    Read testcase_generator.py source, replace line 'N = <int>' with 'N = {N}',
    optionally set seed if the generator uses 'random', then exec in a fresh namespace.

    Returns (N0, edges) where:
      - N0 is max vertex id + 1 inferred from edges (for safety)
      - edges is list of (u, v, w)
    """
    gen_path = os.path.join(os.path.dirname(__file__), "testcase_generator.py")
    with open(gen_path, "r", encoding="utf-8") as f:
        src = f.read()

    # N 치환 (첫 번째 N=정수 선언만 치환)
    src = re.sub(r"^\s*N\s*=\s*\d+\s*$", f"N = {N}", src, count=1, flags=re.MULTILINE)

    # seed를 생성기에 넘기고 싶다면, random.seed 호출 주입(이미 있으면 무시)
    if "random.seed(" not in src:
        src = "import random\nrandom.seed({})\n".format(seed) + src

    # 별도 네임스페이스에서 실행
    ns: dict = {}
    code = compile(src, gen_path, "exec")
    exec(code, ns, ns)

    if "adj" not in ns:
        raise RuntimeError("testcase_generator.py must define global 'adj' (list of (u,v,w)).")

    edges = ns["adj"]
    if not edges:
        return 0, []

    # N0: edges로부터 추론
    maxv = 0
    for (u, v, w) in edges:
        if not (isinstance(u, int) and isinstance(v, int)):
            raise TypeError("Edges must be tuples of ints (u,v,w).")
        if not isinstance(w, (int, float)):
            raise TypeError("Edge weight must be int or float.")
        if u > maxv: maxv = u
        if v > maxv: maxv = v
    N0 = maxv + 1
    return N0, edges

# ------------------------------------------------------------
# 2) run_bmssp.py와 동일한 그래프 준비 → transformGraph() 실행
# ------------------------------------------------------------
def build_graph_like_run_bmssp(N0: int, edges: List[Tuple[int,int,int]], start: int = 0):
    """
    Matches the way run_bmssp prepares algorithms.adj before transformGraph().
    - alg.adj: list[set] of (v, w)
    - alg.N, alg.M, alg.start, vertices, dist/depth/pred 초기화
    """
    alg.N = N0
    alg.M = len(edges)
    alg.start = start
    alg.adj = [set() for _ in range(N0)]
    for (u, v, w) in edges:
        alg.adj[u].add((v, w))
    alg.vertices = list(range(alg.N))

    # k, t 초기화 (transformGraph에서 다시 설정되지만 안전상 넣어둠)
    if alg.N > 1:
        lg = math.log2(alg.N)
        alg.k = max(1, int(math.floor(lg ** (1/3))))
        alg.t = max(1, int(math.floor(lg ** (2/3))))
    else:
        alg.k = alg.t = 1

    INF = math.inf
    alg.dist  = [INF] * alg.N
    alg.depth = [INF] * alg.N
    alg.pred  = [-1]  * alg.N
    alg.dist[alg.start]  = 0
    alg.depth[alg.start] = 0

def transform_once():
    """
    Run algorithms.transformGraph() exactly once.
    This converts alg.adj into list[dict], updates N,M,start,k,t,dist,depth,pred.
    """
    alg.transformGraph()
    if alg.start is None or alg.start < 0:
        raise RuntimeError("transformGraph() produced invalid start. Check generator for isolated start.")

# ------------------------------------------------------------
# 3) 각 알고리즘 실행 래퍼 (변환 이후의 그래프에서 수행)
# ------------------------------------------------------------
def run_baseline_dijkstra():
    """
    Runs Dijkstra on the transformed graph in algorithms.py.
    Expects alg.adj to be list[dict] and alg.start to be valid.
    """
    # 변환 후 N에 맞춰 로컬 dist 반환(alg 전역 변경 없음)
    return dijkstra_on_adj_dict(alg.adj, alg.start)

def run_bmssp_top_level():
    """
    Runs BMSSP on the transformed graph, same as top-level call in the paper:
      l = ceil(log2(N)/t), B = (inf,inf,inf), S = {start}
    Resets alg.dist/depth/pred before run so it doesn't get polluted by Dijkstra.
    """
    n = alg.N
    # transformGraph에서 k,t 재세팅됨
    if n > 1:
        l = math.ceil(math.log2(n) / max(1, alg.t))
    else:
        l = 0

    # SSSP state 초기화
    INF = math.inf
    alg.dist  = [INF] * n
    alg.depth = [INF] * n
    alg.pred  = [-1]  * n
    alg.dist[alg.start]  = 0
    alg.depth[alg.start] = 0

    B = (math.inf, math.inf, math.inf)
    S = {alg.start}
    Bp, U = alg.BMSSP(l, B, S)
    return Bp, U

# ------------------------------------------------------------
# 4) 벤치마크 루프
# ------------------------------------------------------------
def benchmark(ns: List[int], repeats: int = 3, seed: int = 12345):
    """
    For each original N in ns:
      - generate edges with testcase_generator.py (with N replaced)
      - prepare graph like run_bmssp
      - run transformGraph() ONCE (exclude transform time)
      - time Dijkstra on transformed graph
      - time BMSSP top-level on transformed graph
    """
    results = []
    for N in ns:
        # 4.1 generator로 edges 생성
        N0, edges = load_edges_with_generator(N, seed=seed)
        if N0 == 0:
            print(f"[N={N}] generator returned empty graph; skipping.")
            continue

        # 4.2 run_bmssp 방식으로 원그래프 세팅
        build_graph_like_run_bmssp(N0, edges, start=0)

        # 4.3 변환(측정에서 제외)
        t0 = time.perf_counter()
        transform_once()
        t1 = time.perf_counter()
        transform_time = t1 - t0

        # 변환 후 N'/start'
        N_trans = alg.N
        # start_trans = alg.start

        # 4.4 baseline Dijkstra (여러 번 반복 평균)
        d_times = []
        for _ in range(repeats):
            gc.collect()
            t0 = time.perf_counter()
            _ = run_baseline_dijkstra()
            t1 = time.perf_counter()
            d_times.append(t1 - t0)
        d_avg = sum(d_times) / len(d_times)

        # 4.5 BMSSP top-level (여러 번 반복 평균)
        b_times = []
        last_U = None                     # ← 추가
        for _ in range(repeats):
            gc.collect()
            t0 = time.perf_counter()
            _Bp, _U = run_bmssp_top_level()
            last_U = _U                   # ← 추가
            t1 = time.perf_counter()
            b_times.append(t1 - t0)
        b_avg = sum(b_times) / len(b_times)

        # 성공 여부 판단
        success = (last_U is not None and len(last_U) == alg.N)   # ← 추가
        status  = "successful" if success else f"incomplete |U|={len(last_U) if last_U is not None else 'NA'}/{alg.N}"  # ← 추가

        print(f"[N={N}] transform N'={N_trans}| "
              f"transform={transform_time:.3f}s, dijkstra={d_avg:.3f}s, bmssp={b_avg:.3f}s {status}")

        results.append({
            "N": N,
            "N_trans": N_trans,
            "transform_s": transform_time,
            "dijkstra_s": d_avg,
            "bmssp_s": b_avg,
        })
    return results

# ------------------------------------------------------------
# 5) 플로팅/출력
# ------------------------------------------------------------
def plot_results(results, title="Runtime vs N"):
    Ns = [r["N"] for r in results]
    dj = [r["dijkstra_s"] for r in results]
    bm = [r["bmssp_s"] for r in results]

    plt.figure()
    plt.plot(Ns, dj, marker="o", label="Dijkstra (on transformed)")
    plt.plot(Ns, bm, marker="o", label="BMSSP top-level (on transformed)")
    plt.xlabel("Original N (before transform)")
    plt.ylabel("Runtime (s)")
    plt.title(title)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    # 저장은 현재 디렉토리에만 (수동으로 가져가기 용이하게)
    plt.savefig("benchmark_runtime.png", dpi=150)
    plt.show()

def print_table(results):
    print("\nN, N', transform_s, dijkstra_s, bmssp_s")
    for r in results:
        print("{:>8}, {:>8}, {:>9.3f}, {:>10.3f}, {:>9.3f}".format(
            r["N"], r["N_trans"], r["transform_s"], r["dijkstra_s"], r["bmssp_s"]
        ))

if __name__ == "__main__":
    # 필요시 N 리스트 조정 가능
    ns = [2**i for i in range(10, 21)] # 1024 to 2097152
    results = benchmark(ns, repeats=3, seed=12345)
    print_table(results)
    if results:
        plot_results(results, title="Runtime vs N")

