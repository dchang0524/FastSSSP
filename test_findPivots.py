# test_transform_findPivots_connected.py

import random
import math
# ← adjust this to import where you’ve defined transformGraph & findPivots:
import algorithms as m  

def make_random_connected_graph(N0, max_extra_deg=2, wmax=10):
    """
    Returns adj0 as a list of sets of (v, w) tuples:
      - First builds a random spanning tree to ensure connectivity.
      - Then adds up to max_extra_deg extra edges per node.
    """
    nodes = list(range(N0))
    random.shuffle(nodes)

    # use sets of (neighbor, weight)
    adj0 = [ set() for _ in range(N0) ]

    # 1) random spanning tree
    connected = {nodes[0]}
    for v in nodes[1:]:
        u = random.choice(list(connected))
        connected.add(v)
        w = random.randint(1, wmax)
        adj0[u].add((v, w))

    # 2) add a few extra random edges
    for u in range(N0):
        extras = random.randint(0, max_extra_deg)
        choices = [v for v in range(N0) if v != u and all(nbr!=v for nbr,_ in adj0[u])]
        for v in random.sample(choices, min(extras, len(choices))):
            w = random.randint(1, wmax)
            adj0[u].add((v, w))

    return adj0

def test_connected(TRIALS=20):
    for i in range(TRIALS):
        # 1) build original
        N0   = random.randint(5, 15)
        adj0 = make_random_connected_graph(N0)

        # 2) dump into globals
        m.N        = N0
        m.M        = sum(len(s) for s in adj0)
        m.start    = random.randrange(N0)
        m.vertices = list(range(N0))
        m.adj      = [s.copy() for s in adj0]
        m.dist     = [math.inf]*N0
        m.depth    = [math.inf]*N0
        m.pred     = [-1]*N0

        # 3) transform (no args)
        m.transformGraph()

        # 4) capture transformed globals
        N, adj, start = m.N, m.adj, m.start
        dist, depth   = m.dist, m.depth
        k             = m.k

        # 5) run findPivots on transformed graph
        B = math.inf
        P, W = m.findPivots(B, {start})

        # 6) invariants
        assert {start}.issubset(W), \
            f"[{i}] start ∉ W: {start} vs {W}"
        assert len(P)*k <= len(W), \
            f"[{i}] |P|*k > |W|: |P|={len(P)}, k={k}, |W|={len(W)}"
        assert all(dist[v] < B for v in W), \
            f"[{i}] some v∈W has dist[v] ≥ B"

    print(f"✅ All {TRIALS} connected‐graph tests passed.")

if __name__ == "__main__":
    test_connected()
