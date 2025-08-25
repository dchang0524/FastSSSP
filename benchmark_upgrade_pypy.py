# benchmark_upgrade_pypy.py
# PyPy-friendly benchmark runner: no matplotlib, no external plotting.
# - Uses testcase_generator (your real generator):
#   * Prefer testcase_generator.build_graph(N, seed) if it exists.
#   * Otherwise: read testcase_generator.py source, patch "N = ..." and seed,
#     exec it, and use its global 'adj' (and optional 'start').
# - Falls back to an internal generator only if testcase_generator is missing.
#
# Outputs: prints a summary and writes CSV "pypy_results.csv"

from __future__ import annotations
import time
import csv
import math
import random
import importlib
import importlib.util
import re
from typing import List, Tuple, Dict

import algorithms as alg              # your BMSSP + transformGraph
import dijkstra_baseline as dijk      # your baseline Dijkstra on dict adjacency

# ---------- helpers ----------

Edge = Tuple[int, int, int]  # (u, v, w)


def edges_to_adj_set(N: int, edges: List[Edge]) -> List[set]:
    """Adjacency for transformGraph(): list[set[(v, w)]], indexed by u."""
    adj = [set() for _ in range(N)]
    for u, v, w in edges:
        adj[u].add((v, w))
    return adj


def edges_to_adj_dict(N: int, edges: List[Edge]) -> List[Dict[int, int]]:
    """Adjacency for Dijkstra: list[dict[v]=w], indexed by u."""
    adj = [dict() for _ in range(N)]
    for u, v, w in edges:
        if v not in adj[u] or w < adj[u][v]:
            adj[u][v] = w
    return adj


# ---------- internal fallback generator (only if testcase_generator missing) ----------

def build_graph_fallback(N: int, seed: int = 0) -> Tuple[int, int, List[Edge]]:
    rnd = random.Random(seed)
    edges: List[Edge] = []
    for i in range(1, N):
        j = rnd.randrange(0, i)
        w = rnd.randint(1, 20)
        edges.append((j, i, w))
        edges.append((i, j, w))
    extra = 2 * N
    existing = set((u, v) for (u, v, _) in edges)
    while extra > 0:
        u = rnd.randrange(0, N)
        v = rnd.randrange(0, N - 1)
        if v >= u:
            v += 1
        if (u, v) in existing:
            continue
        w = rnd.randint(1, 20)
        edges.append((u, v, w))
        existing.add((u, v))
        extra -= 1
    start = 0
    return N, start, edges


# ---------- use testcase_generator if present (function OR global adj) ----------

def _load_from_tg_source(N: int, seed: int) -> Tuple[int, int, List[Edge]] | None:
    """
    Read testcase_generator.py source, patch "N = ..." and inject seed, exec it,
    then pull 'adj' (and optional 'start').
    Returns None if anything fails.
    """
    try:
        spec = importlib.util.find_spec("testcase_generator")
        if not spec or not spec.origin:
            return None
        with open(spec.origin, "r", encoding="utf-8") as f:
            src = f.read()

        # Patch the first top-level "N = <int>" to desired N
        src2, n_sub = re.subn(r"(?m)^\s*N\s*=\s*\d+\s*$", f"N = {N}", src, count=1)

        # Ensure deterministic seeding: after first "import random"
        if "random.seed(" not in src2:
            src2 = re.sub(r"(?m)^\s*import\s+random\s*$",
                          f"import random\nrandom.seed({seed})",
                          src2, count=1)

        ns: Dict[str, object] = {}
        exec(src2, ns, ns)

        if "adj" not in ns:
            return None
        edges_obj = ns["adj"]
        # Allow either list of (u,v,w) triples or set/iterable
        edges = [(int(u), int(v), int(w)) for (u, v, w) in edges_obj]
        N0 = int(ns.get("N", N))
        start0 = int(ns.get("start", 0))
        return N0, start0, edges
    except Exception:
        return None


def load_edges_with_generator(N: int, seed: int = 0) -> Tuple[int, int, List[Edge]]:
    """
    Priority:
      1) testcase_generator.build_graph(N, seed) if available.
      2) Else testcase_generator module's global 'adj' via source patch/exec (so N follows our request).
      3) Else fallback generator.
    """
    try:
        tg = importlib.import_module("testcase_generator")
    except Exception:
        # Module missing -> fallback
        return build_graph_fallback(N, seed)

    # Case 1: function exists
    build_fn = getattr(tg, "build_graph", None)
    if callable(build_fn):
        out = build_fn(N, seed)
        if len(out) == 3:
            N0, start0, edges0 = out
        elif len(out) == 2:
            start0, edges0 = out
            N0 = N
        else:
            raise RuntimeError("testcase_generator.build_graph must return (N, start, edges) or (start, edges)")
        return int(N0), int(start0), list(edges0)

    # Case 2: no function -> read/patch/exec the source and pull 'adj'
    patched = _load_from_tg_source(N, seed)
    if patched is not None:
        return patched

    # Case 3: fallback
    return build_graph_fallback(N, seed)


# ---------- runners ----------

def run_transform_then_dijkstra_then_bmssp(N0: int, start0: int, edges0: List[Edge]):
    # set original graph for transformGraph (list[set])
    alg.N = N0
    alg.adj = edges_to_adj_set(N0, edges0)
    alg.start = start0
    alg.vertices = list(range(alg.N))

    # transform
    t0 = time.perf_counter()
    alg.transformGraph()
    t1 = time.perf_counter()

    N1 = alg.N
    start1 = alg.start
    adj_dict = alg.adj  # list[dict[v]=w]

    # Dijkstra on transformed graph
    t2 = time.perf_counter()
    _ = dijk.dijkstra_on_adj_dict(adj_dict, start1)
    t3 = time.perf_counter()

    # BMSSP on transformed graph
    l = math.ceil(math.log2(N1) / max(1, alg.t)) if N1 > 1 else 0
    t4 = time.perf_counter()
    B_prime, U = alg.BMSSP(l, (math.inf, math.inf, math.inf), {alg.start})
    t5 = time.perf_counter()

    status = "successful" if len(U) == N1 else "partial"

    return {
        "N0": N0,
        "N1": N1,
        "start1": start1,
        "t_transform": t1 - t0,
        "t_dijkstra": t3 - t2,
        "t_bmssp": t5 - t4,
        "ratio": (t5 - t4) / (t3 - t2) if (t3 - t2) > 0 else float("inf"),
        "status": status,
    }


def benchmark(ns: List[int], repeats: int = 3, seed: int = 0, csv_path: str = "pypy_results.csv"):
    rows = []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "N'", "transform_s", "dijkstra_s", "bmssp_s", "ratio", "status"])

    for N in ns:
        Ngen, start0, edges = load_edges_with_generator(N, seed=seed)

        best = None
        for _ in range(repeats):
            stats = run_transform_then_dijkstra_then_bmssp(Ngen, start0, edges)
            if best is None or stats["t_bmssp"] < best["t_bmssp"]:
                best = stats

        print(f"[N={best['N0']}] transform N'={best['N1']}| "
              f"transform={best['t_transform']:.3f}s, "
              f"dijkstra={best['t_dijkstra']:.3f}s, "
              f"bmssp={best['t_bmssp']:.3f}s "
              f"{best['status']}")

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([best["N0"], best["N1"], f"{best['t_transform']:.3f}",
                        f"{best['t_dijkstra']:.3f}", f"{best['t_bmssp']:.3f}",
                        f"{best['ratio']:.2f}", best["status"]])
        rows.append(best)

    print("\n=== Summary (times) ===")
    print("N, N', transform_s, dijkstra_s, bmssp_s, ratio, status")
    for r in rows:
        print(f"{r['N0']:8d}, {r['N1']:8d}, {r['t_transform']:.3f}, "
              f"{r['t_dijkstra']:.3f}, {r['t_bmssp']:.3f}, {r['ratio']:.2f}, {r['status']}")


if __name__ == "__main__":
    ns = [2**17, 2**18, 2**19, 2**20, 2**21]  # adjust freely
    benchmark(ns, repeats=3, seed=12345)

