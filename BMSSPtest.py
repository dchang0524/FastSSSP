
import math
import traceback
from pprint import pprint

# Import user's modules
import algorithms as alg
import D as Dmod

# ---------- Global watchdogs ----------
MAX_CALLS = 2000
MAX_DEPTH = 200
_call_count = 0
_stack = []

def fmt_tuple(t):
    try:
        # distance, depth, u, v (some may be missing depending on site of insert)
        if isinstance(t, tuple):
            return tuple(t)
    except Exception:
        pass
    return t

# ---------- Debug DataStructure wrapper ----------
# --- READ-ONLY debugger for D ---
class DebugDataStructureD(Dmod.DataStructureD):
    # ---------- Helpers (read-only) ----------
    def _dump_blocks(self, label="STATE"):
        # D0
        print(f"  ┌── {label}: D0 ──")
        blk = self.d0_head
        idx = 0
        while blk:
            vals = []
            n = blk.head
            while n:
                vals.append((n.key, n.value))
                n = n.next
            print(f"    D0[{idx}] bound={blk.bound}, size={blk.size}, items={vals}")
            blk = blk.next_block
            idx += 1

        # D1
        print(f"  ┌── {label}: D1 ──")
        for bnd, blk in self.D1.items():
            vals = []
            n = blk.head
            while n:
                vals.append((n.key, n.value))
                n = n.next
            print(f"    bound={bnd}, size={blk.size}, items={vals}")
        print("  └────────")


    def _compute_collected_blocks(self):
        """Compute which blocks would be collected (D0 prefix then D1) without mutating."""
        collected, count = [], 0
        cur = self.d0_head
        while cur:
            collected.append(cur)
            count += cur.size
            if count >= self.M:
                return collected
            cur = cur.next_block
        if count < self.M:
            for _, blk in self.D1.items():
                collected.append(blk)
                count += blk.size
                if count >= self.M:
                    break
        return collected

    def _dump_collected(self, collected):
        print("  ┌── Collected blocks (preview) ──")
        for i, blk in enumerate(collected):
            vals = []
            n = blk.head
            while n:
                vals.append((n.key, n.value))
                n = n.next
            print(f"    collected[{i}] bound={blk.bound}, size={blk.size}, items={vals}")
        print("  └────────")

    # ---------- Wrapped methods (call super, then dump) ----------
    def insert(self, key, value_tuple):
        # pre: do not touch internal state
        print(f"[D.insert] key={key}, value={value_tuple}, nnz(before)={self.nnz}")
        super().insert(key, value_tuple)  # actual mutation happens here
        print(f"[D.insert] nnz(after)={self.nnz}")
        # post: read-only dump
        #self._dump_blocks("after insert")

    def batch_prepend(self, items):
        print(f"[D.batch_prepend] adding {len(items)} items, nnz(before)={self.nnz}")
        # IMPORTANT: do not iterate or transform 'items' here; just pass through
        super().batch_prepend(items)
        print(f"[D.batch_prepend] nnz(after)={self.nnz}")
        #self._dump_blocks("after batch_prepend")

    def pull(self):
        # pre-state dump (read-only)
        print(f"[D.pull] start, nnz={self.nnz}, M={self.M}")
        #self._dump_blocks("before pull")
        #collected = self._compute_collected_blocks()
        #self._dump_collected(collected)

        # call real pull (mutates, removes up to M items)
        res = super().pull()

        # post-state + result
        # if res is None:
        #     print("[D.pull] returned None")
        #     return None
        B_i, S_i = res  # DO NOT reorder / modify!
        try:
            sample = list(S_i)[:8]
        except Exception:
            sample = f"<type {type(S_i).__name__}>"
        print(f"[D.pull] -> Bi={B_i}, |Si|={len(S_i)}, sample={sample}")
        #self._dump_blocks("after pull")
        return res


# Monkeypatch algorithms to use debug D
_orig_DataStructureD = alg.DataStructureD
alg.DataStructureD = DebugDataStructureD

# Save originals
_orig_BMSSP = alg.BMSSP
_orig_BaseCase = alg.BaseCase
_orig_findPivots = alg.findPivots

def _guard():
    global _call_count
    _call_count += 1
    if _call_count > MAX_CALLS:
        raise RuntimeError(f"BMSSP call count exceeded {MAX_CALLS} (possible infinite recursion).")

# ---------- Debug wrappers ----------
def Debug_BaseCase(B, S):
    print(f"[BaseCase] Enter with B={B}, S={S}")
    out = _orig_BaseCase(B, S)
    print(f"[BaseCase] Exit -> B'={out[0]}, |U|={len(out[1])}, U={sorted(list(out[1]))}")
    return out

def Debug_findPivots(B, S):
    print(f"[findPivots] Enter with B={B}, |S|={len(S)}")
    P, W = _orig_findPivots(B, S)
    print(f"[findPivots] Exit -> |P|={len(P)}, |W|={len(W)}  (P sample={list(P)[:5]}, W sample={list(W)[:5]})")
    return P, W

def Debug_BMSSP(l, B, S):
    _guard()
    depth = len(_stack)
    indent = "  " * depth
    print(f"{indent}[BMSSP] ► Enter l={l}, B={B}, |S|={len(S)}, S(sample)={list(S)[:6]}")
    _stack.append((l, B, set(S)))
    try:
        out = _orig_BMSSP(l, B, S)
        Bp, U = out
        print(f"{indent}[BMSSP] ◄ Exit  l={l}, returns B'={Bp}, |U|={len(U)}, U(sample)={sorted(list(U))[:10]}")
        return out
    except Exception as e:
        print(f"{indent}[BMSSP] !! Exception at l={l}: {e}")
        traceback.print_exc()
        raise
    finally:
        _stack.pop()

# Apply monkeypatches
alg.BaseCase = Debug_BaseCase
alg.findPivots = Debug_findPivots
alg.BMSSP = Debug_BMSSP

# ---------- Graph builders ----------
import random

def make_reachable_random_digraph(
    N: int,
    *,
    max_wt: int = 10,
    extra_prob: float | None = 0.1,   # use either this...
    extra_edges: int | None = None,   # ...or this (count), not both
    seed: int | None = None,
) -> list[set[tuple[int, int]]]:
    """
    Returns adjacency list: graph[u] is a set of (v, w) pairs.
    Guarantees every node is reachable from 0 via a directed tree,
    then adds random extra edges.
    """
    assert N >= 1
    if seed is not None:
        random.seed(seed)

    g = [set() for _ in range(N)]

    # 1) Build a random arborescence rooted at 0 (u -> v edges)
    # Pick a random parent among [0 .. v-1] to ensure reachability (and acyclicity for the tree edges).
    for v in range(1, N):
        p = random.randrange(0, v)
        w = random.randint(1, max_wt)
        g[p].add((v, w))

    # Helper to add an edge if it doesn't already exist
    def add_edge(u, v):
        if u == v:
            return False
        if any(x == v for x, _ in g[u]):
            return False
        w = random.randint(1, max_wt)
        g[u].add((v, w))
        return True

    # 2) Add extra random edges
    if extra_edges is not None:
        # Add exactly this many successful extra edges (best-effort)
        tries = 0
        needed = extra_edges
        max_tries = extra_edges * 20 + 100
        while needed > 0 and tries < max_tries:
            u = random.randrange(0, N)
            v = random.randrange(0, N)
            if add_edge(u, v):
                needed -= 1
            tries += 1
    elif extra_prob is not None and extra_prob > 0:
        # Probabilistic sprinkling
        for u in range(N):
            for v in range(N):
                if u == v:
                    continue
                if random.random() < extra_prob:
                    add_edge(u, v)

    return g
    
def reset_graph(n):
    # Reset globals in algorithms
    alg.N = n
    alg.vertices = list(range(n))
    alg.adj = [dict() for _ in range(n)]
    alg.dist = [math.inf] * n
    alg.depth = [math.inf] * n
    alg.pred = [-1] * n
    # recompute k,t based on log2(N) per user's code
    from math import log2
    alg.k = max(2, math.floor(log2(n) ** (1/3))) if n > 1 else 1
    alg.t = math.floor(log2(n) ** (2/3)) if n > 1 else 1
    # if alg.k < 2: alg.k = 2
    # if alg.t < 2: alg.t = 2
    # watchdog reset
    global _call_count, _stack
    _call_count = 0
    _stack = []

def add_edge(u, v, w):
    alg.adj[u][v] = w

def init_source(s=0):
    alg.start = s
    alg.dist[s] = 0.0
    alg.depth[s] = 0
    alg.pred[s] = s

def line_graph(n, w=1.0):
    reset_graph(n)
    for i in range(n-1):
        add_edge(i, i+1, w)
    init_source(0)

def cycle_with_chord(n):
    reset_graph(n)
    # simple cycle
    for i in range(n):
        add_edge(i, (i+1)%n, 1.0)
    # add a chord to create alternative route
    add_edge(0, n//2, 0.3)
    init_source(0)

def diamond_graph():
    reset_graph(6)
    # 0 -> 1 -> 3 -> 5
    #  \-> 2 -> 4 -/
    add_edge(0,1,1.0); add_edge(1,3,1.0); add_edge(3,5,1.0)
    add_edge(0,2,1.0); add_edge(2,4,1.0); add_edge(4,5,1.0)
    init_source(0)

def branched_graph():
    reset_graph(8)
    # Tree-ish with back edges
    add_edge(0,1,1.0); add_edge(0,2,2.0)
    add_edge(1,3,1.0); add_edge(1,4,2.5)
    add_edge(2,5,0.5); add_edge(5,6,0.5); add_edge(6,4,0.5)
    add_edge(4,7,1.0)
    init_source(0)

def run_test(name, builder, l=None, B=None):
    print("\n" + "="*80)
    print(f"TEST: {name}")
    builder()
    n = alg.N
    # choose l default
    if l is None:
        # depth ~ log n / t, pick 2 as safe small
        l = math.ceil(math.log2(n) / alg.t)
    if B is None:
        B = (math.inf, float('inf'), float('inf')) if isinstance(alg.dist[0], tuple) else math.inf
    # S = {source}
    S = {alg.start}
    print(f"Params: l={l}, B={B}, S={S}, k={alg.k}, t={alg.t}")
    try:
        Bp, U = alg.BMSSP(l, B, S)
        print(f"[RESULT] B'={Bp}, |U|={len(U)}, U(sorted)={sorted(list(U))}")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    # Run a suite
    run_test("Line(6)", lambda: line_graph(6))
    run_test("CycleWithChord(8)", lambda: cycle_with_chord(8))
    run_test("Diamond", diamond_graph)
    #run_test("Branched", branched_graph, l=3)