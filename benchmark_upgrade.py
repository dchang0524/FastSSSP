# benchmark_compare.py
import os
import re
import math
import time
import gc
from typing import List, Tuple
import matplotlib.pyplot as plt
import json

# 1) import your code as usual
import algorithms as alg
# 2) apply the profiler/ops-count patch in-place
import alg_patch  # <-- NEW: patches 'alg' module symbols

from dijkstra_baseline import dijkstra_on_adj_dict

# ------------------------------------------------------------
# (unchanged) generator
# ------------------------------------------------------------
def load_edges_with_generator(N: int, seed: int = 12345) -> Tuple[int, list[tuple[int,int,int]]]:
    gen_path = os.path.join(os.path.dirname(__file__), "testcase_generator.py")
    with open(gen_path, "r", encoding="utf-8") as f:
        src = f.read()
    src = re.sub(r"^\s*N\s*=\s*\d+\s*$", f"N = {N}", src, count=1, flags=re.MULTILINE)
    if "random.seed(" not in src:
        src = "import random\nrandom.seed({})\n".format(seed) + src
    ns: dict = {}
    code = compile(src, gen_path, "exec")
    exec(code, ns, ns)
    if "adj" not in ns:
        raise RuntimeError("testcase_generator.py must define global 'adj' (list of (u,v,w)).")
    edges = ns["adj"]
    if not edges:
        return 0, []
    maxv = 0
    for (u, v, w) in edges:
        if u > maxv: maxv = u
        if v > maxv: maxv = v
    return maxv + 1, edges

# ------------------------------------------------------------
# (unchanged) build graph like run_bmssp
# ------------------------------------------------------------
def build_graph_like_run_bmssp(N0: int, edges: list[tuple[int,int,int]], start: int = 0):
    alg.N = N0
    alg.M = len(edges)
    alg.start = start
    alg.adj = [set() for _ in range(N0)]
    for (u, v, w) in edges:
        alg.adj[u].add((v, w))
    alg.vertices = list(range(alg.N))
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
    alg.transformGraph()
    if alg.start is None or alg.start < 0:
        raise RuntimeError("transformGraph() produced invalid start. Check generator for isolated start.")

# ------------------------------------------------------------
# (unchanged) runners
# ------------------------------------------------------------
def run_baseline_dijkstra():
    return dijkstra_on_adj_dict(alg.adj, alg.start)

def run_bmssp_top_level():
    n = alg.N
    l = math.ceil(math.log2(n) / max(1, alg.t)) if n > 1 else 0
    INF = math.inf
    alg.dist  = [INF] * n
    alg.depth = [INF] * n
    alg.pred  = [-1]  * n
    alg.dist[alg.start]  = 0
    alg.depth[alg.start] = 0
    B = (math.inf, math.inf, math.inf)
    S = {alg.start}
    return alg.BMSSP(l, B, S)

# ------------------------------------------------------------
# helpers for pretty print
# ------------------------------------------------------------
KEYS_TIME = [
    "time.transform", "time.findPivots", "time.BaseCase", "time.BMSSP",
    "time.D.insert", "time.D.pull", "time.D.batch_prepend",
]
KEYS_COUNT = [
    "count.findPivots.calls", "count.BaseCase.calls", "count.BMSSP.calls",
    "count.BMSSP.iterations", "count.relax_attempt", "count.needsUpdate",
    "count.update", "count.relax_success",
    "count.heap_push", "count.heap_pop",
    "count.W_total", "count.P_selected",
    "count.D.insert", "count.D.pull", "count.D.batch_prepend", "count.batch_prepend.items",
]

def prof_snapshot():
    return alg.get_prof()

def prof_reset():
    alg.reset_prof()

# ------------------------------------------------------------
# benchmark with profile snapshots
# ------------------------------------------------------------
def benchmark(ns: list[int], repeats: int = 1, seed: int = 12345):
    results = []
    for N in ns:
        N0, edges = load_edges_with_generator(N, seed=seed)
        if N0 == 0:
            print(f"[N={N}] generator returned empty graph; skipping.")
            continue

        build_graph_like_run_bmssp(N0, edges, start=0)

        # Transform (profile separately)
        prof_reset()
        t0 = time.perf_counter()
        transform_once()
        t1 = time.perf_counter()
        transform_time = t1 - t0
        prof_transform = prof_snapshot()
        prof_reset()

        N_trans = alg.N

        # Dijkstra
        d_times = []
        for _ in range(repeats):
            gc.collect()
            t0 = time.perf_counter()
            _ = run_baseline_dijkstra()
            t1 = time.perf_counter()
            d_times.append(t1 - t0)
        d_avg = sum(d_times) / len(d_times)

        # BMSSP
        prof_reset()
        b_times = []
        last_U = None
        for _ in range(repeats):
            gc.collect()
            t0 = time.perf_counter()
            _Bp, _U = run_bmssp_top_level()
            last_U = _U
            t1 = time.perf_counter()
            b_times.append(t1 - t0)
        b_avg = sum(b_times) / len(b_times)
        prof_bmssp = prof_snapshot()

        success = (last_U is not None and len(last_U) == alg.N)
        status  = "successful" if success else f"incomplete |U|={len(last_U) if last_U is not None else 'NA'}/{alg.N}"

        print(f"[N={N}] transform N'={N_trans}| "
              f"transform={transform_time:.3f}s, dijkstra={d_avg:.3f}s, bmssp={b_avg:.3f}s {status}")
        results.append({
            "N": N,
            "N_trans": N_trans,
            "transform_s": transform_time,
            "dijkstra_s": d_avg,
            "bmssp_s": b_avg,
            "prof_transform": prof_transform,
            "prof_bmssp": prof_bmssp,
            "status": status,
        })
    return results

def plot_results(results, title="Runtime ratio: BMSSP / Dijkstra"):
    Ns = [r["N"] for r in results]
    dj = [r["dijkstra_s"] for r in results]
    bm = [r["bmssp_s"] for r in results]
    ratio = [ (b/d) if d > 0 else float("inf") for b, d in zip(bm, dj) ]
    plt.figure()
    plt.plot(Ns, ratio, marker="o", label="BMSSP / Dijkstra")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Original N (before transform)")
    plt.ylabel("Runtime ratio (T_BMSSP / T_Dijkstra)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_ratio.png", dpi=150)
    plt.show()

def _fmt_time(prof):
    return ", ".join(f"{k.split('.',1)[1]}={prof.get(k,0.0):.3f}s" for k in KEYS_TIME)

def _fmt_counts(prof):
    return ", ".join(f"{k.split('.',1)[1]}={prof.get(k,0)}" for k in KEYS_COUNT)

def print_table(results):
    print("\n=== Summary (times) ===")
    print("N, N', transform_s, dijkstra_s, bmssp_s, ratio, status")
    for r in results:
        ratio = (r["bmssp_s"] / r["dijkstra_s"]) if r["dijkstra_s"] > 0 else float("inf")
        print("{:>8}, {:>8}, {:>9.3f}, {:>10.3f}, {:>9.3f}, {:>7.2f}, {}".format(
            r["N"], r["N_trans"], r["transform_s"], r["dijkstra_s"], r["bmssp_s"], ratio, r["status"]
        ))

    print("\n=== Transform profile (time only) ===")
    for r in results:
        print(f"N={r['N']} -> {_fmt_time(r['prof_transform'])}")

    print("\n=== BMSSP profile (time) ===")
    for r in results:
        print(f"N={r['N']} -> {_fmt_time(r['prof_bmssp'])}")

    print("\n=== BMSSP profile (counts) ===")
    for r in results:
        print(f"N={r['N']} -> {_fmt_counts(r['prof_bmssp'])}")

def dump_profiles(results, fname="benchmark_profiles.json"):
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    ns = [2**i for i in range(17, 22)]  # 131072 to 2097152
    results = benchmark(ns, repeats=1, seed=12345)
    print_table(results)
    dump_profiles(results, "benchmark_profiles.json")
    if results:
        plot_results(results, title="Runtime ratio: BMSSP / Dijkstra")
