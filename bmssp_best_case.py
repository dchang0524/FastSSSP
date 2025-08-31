# benchmark_compare_bmssp_ideal.py
import os
import math
import time
import gc
from typing import List, Tuple

# Local modules
import algorithms as alg            # your BMSSP / transformGraph / BMSSP(l,B,S)
from dijkstra_baseline import dijkstra_on_adj_dict


# ------------------------------------------------------------
# 1) Build the "ideal for BMSSP" topology programmatically
#    Topology:
#      s -> A-layer (~sqrt(n))
#      A(i) -> block of B-layer nodes; total |B| ~= n (wide)
#      every B -> each small hub in H (|H| = h, tiny)
#      hubs -> a skinny chain (rest) so all long paths pass H
#
#    N_param = target "n" for the wide middle (B layer)
#    Returns (N0, edges) with vertex ids in [0..N0-1]
# ------------------------------------------------------------
def build_bmssp_ideal_edges(
    N_param: int,
    hub_count: int = 3,
    chain_len: int | None = None,
    weight: int = 1
) -> Tuple[int, List[Tuple[int, int, int]]]:
    """
    Construct directed edges for the 'ideal' BMSSP graph.

    Vertex layout:
      0                 -> source s
      [1 .. A]          -> first narrow layer (size A ~ floor(sqrt(n)))
      [A+1 .. A+B]      -> massive wide middle layer (size B ~ n)
      [A+B+1 .. A+B+h]  -> tiny hub set H (size h=hub_count)
      [..]              -> skinny chain (size R = chain_len or ~A)

    All edges have the same positive weight.
    """
    # Guard rails
    n = max(1, int(N_param))       # middle layer target
    h = max(1, int(hub_count))     # tiny hubs
    # First layer size ~ sqrt(n)
    A = max(1, int(math.isqrt(n)))
    # Make B as close to n as possible (<= A*block, but we’ll fill exactly n nodes)
    B = n
    # Reasonable tail length if unspecified: proportional to A
    if chain_len is None:
        R = max(1, A)
    else:
        R = max(1, int(chain_len))

    # Assign ids
    s = 0
    a_start = 1
    a_end   = a_start + A - 1

    b_start = a_end + 1
    b_end   = b_start + B - 1

    h_start = b_end + 1
    h_end   = h_start + h - 1

    r_start = h_end + 1
    r_end   = r_start + R - 1

    N0 = r_end + 1  # total vertices
    edges: List[Tuple[int, int, int]] = []

    # s -> A-layer
    for u in range(a_start, a_end + 1):
        edges.append((s, u, weight))

    # Partition B (~n) fairly evenly among A parents
    # Each A(i) connects to a block of B nodes
    block_size = (B + A - 1) // A  # ceil division
    b_ptr = b_start
    for i in range(A):
        # block [b_ptr .. b_ptr + block_size - 1], but not exceed b_end
        blk_end = min(b_ptr + block_size - 1, b_end)
        if b_ptr <= blk_end:
            a_node = a_start + i
            for v in range(b_ptr, blk_end + 1):
                edges.append((a_node, v, weight))
            b_ptr = blk_end + 1
        if b_ptr > b_end:
            break

    # Wide middle -> hubs: each B node connects to each hub
    for u in range(b_start, b_end + 1):
        for hv in range(h_start, h_end + 1):
            edges.append((u, hv, weight))

    # Hubs -> skinny chain
    if R == 1:
        # trivial one-node tail; connect hubs to it
        if r_start <= r_end:
            tail0 = r_start
            for hv in range(h_start, h_end + 1):
                edges.append((hv, tail0, weight))
    else:
        # hubs -> first chain node
        first_tail = r_start
        for hv in range(h_start, h_end + 1):
            edges.append((hv, first_tail, weight))
        # chain self: r_i -> r_{i+1}
        for u in range(r_start, r_end):
            edges.append((u, u + 1, weight))

    return N0, edges


# ------------------------------------------------------------
# 2) Prepare graph exactly like run_bmssp does, then transform()
# ------------------------------------------------------------
def build_graph_like_run_bmssp(N0: int, edges: List[Tuple[int, int, int]], start: int = 0):
    """
    Matches the way run_bmssp prepares algorithms.adj before transformGraph().
    - alg.adj: list[set] of (v, w)
    - alg.N, alg.M, alg.start, etc.
    """
    alg.N = N0
    alg.M = len(edges)
    alg.start = start
    alg.adj = [set() for _ in range(N0)]
    for (u, v, w) in edges:
        alg.adj[u].add((v, w))
    alg.vertices = list(range(alg.N))

    # Pre-init (transformGraph may recompute)
    if alg.N > 1:
        lg = math.log2(alg.N)
        alg.k = max(1, int(math.floor(lg ** (1 / 3))))
        alg.t = max(1, int(math.floor(lg ** (2 / 3))))
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
    Run algorithms.transformGraph() once (exclude time from core timings).
    """
    alg.transformGraph()
    if alg.start is None or alg.start < 0:
        raise RuntimeError("transformGraph() produced invalid start; check constructed graph.")


# ------------------------------------------------------------
# 3) Algorithm runners on transformed graph
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
    Bp, U = alg.BMSSP(l, B, S)
    return Bp, U


# ------------------------------------------------------------
# 4) Benchmark loop (same shape/prints as your script)
#    ns = list of 'n' (size of the wide middle layer)
# ------------------------------------------------------------
def benchmark(ns: List[int], repeats: int = 1, hub_count: int = 3, chain_len: int | None = None):
    results = []
    for n_mid in ns:
        # 4.1 construct ideal edges
        N0, edges = build_bmssp_ideal_edges(
            N_param=n_mid,
            hub_count=hub_count,
            chain_len=chain_len,
            weight=1,
        )
        if N0 == 0:
            print(f"[n={n_mid}] empty graph; skipping.")
            continue

        # 4.2 run_bmssp-style setup
        build_graph_like_run_bmssp(N0, edges, start=0)

        # 4.3 transform (excluded from core times)
        t0 = time.perf_counter()
        transform_once()
        t1 = time.perf_counter()
        transform_time = t1 - t0

        N_trans = alg.N

        # 4.4 Dijkstra (average of repeats)
        d_times = []
        for _ in range(repeats):
            gc.collect()
            t0 = time.perf_counter()
            _ = run_baseline_dijkstra()
            t1 = time.perf_counter()
            d_times.append(t1 - t0)
        d_avg = sum(d_times) / len(d_times)

        # 4.5 BMSSP top-level (average of repeats)
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

        success = (last_U is not None and len(last_U) == alg.N)
        status  = "successful" if success else f"incomplete |U|={len(last_U) if last_U is not None else 'NA'}/{alg.N}"

        print(f"[n={n_mid}] transform N'={N_trans}| "
              f"transform={transform_time:.3f}s, dijkstra={d_avg:.3f}s, bmssp={b_avg:.3f}s {status}")

        results.append({
            "N": n_mid,             # keep key names consistent with your printer/plotter
            "N_trans": N_trans,
            "transform_s": transform_time,
            "dijkstra_s": d_avg,
            "bmssp_s": b_avg,
        })
    return results


# ------------------------------------------------------------
# 5) Plotting / Printing (identical interface)
# ------------------------------------------------------------
def plot_results(results, title="Runtime ratio: BMSSP / Dijkstra"):
    Ns = [r["N"] for r in results]
    dj = [r["dijkstra_s"] for r in results]
    bm = [r["bmssp_s"] for r in results]

    ratio = [(b / d) if d > 0 else float("inf") for b, d in zip(bm, dj)]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(Ns, ratio, marker="o", label="BMSSP / Dijkstra")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Middle layer size n (before transform)")
    plt.ylabel("Runtime ratio (T_BMSSP / T_Dijkstra)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_ratio_bmssp_ideal.png", dpi=150)
    plt.show()


def print_table(results):
    print("\nN, N', transform_s, dijkstra_s, bmssp_s, ratio")
    for r in results:
        ratio = (r["bmssp_s"] / r["dijkstra_s"]) if r["dijkstra_s"] > 0 else float("inf")
        print("{:>8}, {:>8}, {:>9.3f}, {:>10.3f}, {:>9.3f}, {:>7.2f}".format(
            r["N"], r["N_trans"], r["transform_s"], r["dijkstra_s"], r["bmssp_s"], ratio
        ))


if __name__ == "__main__":
    # Choose n as the WIDTH of the middle layer (stress Dijkstra’s queue growth).
    # You can tune hub_count and chain_len to your taste.
    ns = [2**i for i in range(14, 21)]  # 16384 to ~2M middle nodes
    results = benchmark(ns, repeats=3, hub_count=3, chain_len=None)
    print_table(results)
    if results:
        plot_results(results, title="BMSSP-ideal graph: Runtime ratio (BMSSP / Dijkstra)")
