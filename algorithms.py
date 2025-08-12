import math
from math import log2
import heapq
# from D import DataStructureD
from lemma33_data_structure import Lemma33DataStructure as DataStructureD

INF = math.inf          # ───────────── sentinel 하나만 둡니다

N = 100000
M = 0
k = math.floor(log2(N) ** (1/3))
t = math.floor(log2(N) ** (2/3))
start = 0
vertices = [i for i in range(N)]
adj = [dict() for _ in range(N)]
dist = [INF] * N  # sum of path weights
depth = [INF] * N   # number of vertices traversed
pred = [-1] * N  #last previous vertex visited in path

#Graph Transformation
def transformGraph():
    global N, M, start, adj, vertices, dist, depth, pred, k, t

    # Keep a reference to the original graph structure to read from
    old_adj = adj
    old_start = start
    old_n = N

    # 1. Find all neighbors (incoming and outgoing) for each vertex.
    all_neighbors = [[] for _ in range(old_n)]
    for u in range(old_n):
        # old_adj is a list of sets of (neighbor, weight) tuples
        for v, _ in old_adj[u]:
            all_neighbors[u].append(v)
            all_neighbors[v].append(u)
    
    # Handle duplicates
    for i in range(old_n):
        all_neighbors[i] = sorted(list(set(all_neighbors[i])))

    # 2. --- Vertex Renumbering Step ---
    # Create new vertices x_uv and map them to new integer IDs.
    node_map = {}
    new_node_counter = 0
    for u in range(old_n):
        for v in all_neighbors[u]:
            if (u, v) not in node_map:
                node_map[(u, v)] = new_node_counter
                new_node_counter += 1
    
    new_n = new_node_counter
    # The new adjacency list will be a list of dictionaries.
    new_adj_list = [{} for _ in range(new_n)]
    new_m = 0

    # 3. --- Cycle Creation Step ---
    # Create cycles with zero-weight edges.
    for u in range(old_n):
        cycle_nodes = all_neighbors[u]
        if len(cycle_nodes) > 1:
            for i in range(len(cycle_nodes)):
                v1 = cycle_nodes[i]
                v2 = cycle_nodes[(i + 1) % len(cycle_nodes)]
                
                node1_id = node_map.get((u, v1))
                node2_id = node_map.get((u, v2))

                if node1_id is not None and node2_id is not None:
                    # Use dictionary assignment for direct access
                    new_adj_list[node1_id][node2_id] = 0
                    new_m += 1

    # 4. --- Original Edge Weight Step ---
    # Add edges corresponding to the original graph's edges.
    for u in range(old_n):
        # old_adj is a list of sets of (neighbor, weight) tuples
        for v, w in old_adj[u]:
            node_uv_id = node_map.get((u, v))
            node_vu_id = node_map.get((v, u))

            if node_uv_id is not None and node_vu_id is not None:
                # Use dictionary assignment for direct access
                new_adj_list[node_uv_id][node_vu_id] = w
                new_m += 1

    # 5. --- Determine New Start Node ---
    new_start = -1
    if old_adj[old_start]:
        try:
            first_neighbor_v, _ = next(iter(old_adj[old_start]))
            new_start = node_map.get((old_start, first_neighbor_v), -1)
        except StopIteration:
            # old_start has no outgoing edges
            pass

    # --- Update Global Variables ---
    N = new_n
    M = new_m
    adj = new_adj_list # adj is now a list of dictionaries
    start = new_start
    vertices = list(range(N))
    k = math.floor(log2(N) ** (1/3))
    t = math.floor(log2(N) ** (2/3))
    # ★ sentinel = ∞ 로 초기화 ★
    INF = math.inf
    dist  = [INF] * N
    depth = [INF] * N
    pred  = [-1]  * N
    dist[start]  = 0
    depth[start] = 0
          

def needsUpdate(u, v):
    if math.isinf(dist[u]):         # ★ 'is' 대신 isinf
        return False
    nd = dist[u] + adj[u][v]  # 새 후보 거리
    dv = dist[v]

    if nd < dv:
        return True
    if nd == dv:
        # 논문 Assumption 2.1에 맞춘 tie-breaking:
        # 1) hop 수(=depth) 더 짧은 경로 우선
        # 2) 그래도 같으면 endpoint id(여기서는 u)로 정렬
        # (원 논문은 (길이, α, 끝점, …) 순의 사전식 정렬을 가정)
        return (depth[u] + 1, u) < (depth[v], pred[v])
    return False

def update(u, v):
    global N, M, start, adj, vertices, dist, depth, pred, k, t
    new_d = dist[u] + adj[u][v]
    # needsUpdate가 True일 때만 호출된다고 가정
    dist[v]  = new_d
    depth[v] = depth[u] + 1
    pred[v]  = u

def cmp(u, v):                          # tie-breaking이 필요하면
    global N, M, start, adj, vertices, dist, depth, pred, k, t
    if dist[u] is INF or dist[v] is INF:
        return 0                       # 미확정 → 비교 안 함
    return (dist[v], depth[v], v) < (dist[u], depth[u], u)

#Algorithm 1
def findPivots(B, S):
    global N, M, start, adj, vertices, dist, depth, pred, k, t
    """
    U': Contains every incomplete vertex with d(v) < B where the shortest path visits some vertex in S
    Finds a set W of size O(k|S|) and a set of pivots P (subset of S) with |P| <= |W|/k
    For every x in U'
        x is in W and complete (dist[x] = d(x))
        or
        The shortest path to x visits some complete vertex in P
    Runs in O(k|W|) = O(min{k^2|S|, k|U'|})
    """

    # L2, L3
    W  = set(S)
    Wm = set(S)                 # W_0

    kS = k * len(S)

    # L4: for i = 1 to k do
    for _ in range(k):
        Wi = set()              # L5

        # L6: for all edges (u,v) with u ∈ W_{i-1} do
        for u in Wm:
            for v, w in adj[u].items():
                nd = dist[u] + w
                # L7: if d̂[u] + w_uv ≤ d̂[v] then
                if needsUpdate(u, v):               # needsUpdate의 ≤ 규칙
                    # tie-break은 needsUpdate 안에 두셨다면:
                    # if needsUpdate(u, v): update(u, v)
                    # 처럼 바꿔도 됩니다.
                    update(u, v)            # L8

                # L9: if d[u] + w_uv < B then
                if nd < B:
                    # L10: W_i ← W_i ∪ {v}
                    Wi.add(v)

        # L11: W ← W ∪ W_i
        W |= Wi

        # L12–L14: if |W| > k|S| then  P ← S ; return P, W
        if len(W) > kS:
            return set(S), W

        # 다음 단계로 전파
        Wm = Wi

    # (여기까지 왔으면 조기 종료 없었음)
    # L15: F ← {(u,v)∈E : u,v∈W, d̂[v] = d̂[u] + w_uv}
    tAdj = {u: [] for u in W}
    for u in W:
        for v, w in adj[u].items():
            if v in W and dist[v] == dist[u] + w:   # 정확히 등호
                tAdj[u].append(v)                   # Assumption 2.1 하에 forest

    # L16: P ← { u ∈ S : u is a root of a tree with ≥ k vertices in F }
    # (후위 순서로 서브트리 크기 세기)
    post, stack, seen = [], list(S), set()
    while stack:
        u = stack.pop()
        if u in seen: 
            continue
        seen.add(u)
        post.append(u)
        for v in tAdj[u]:
            if v not in seen:
                stack.append(v)

    size = {u: 1 for u in W}
    for u in reversed(post):
        for v in tAdj[u]:
            size[u] += size[v]

    P = {u for u in S if u in W and size.get(u, 0) >= k}

    # L17
    return P, W


# Algorithm 2
def BaseCase(B, S):
    """
    Given a bound B and a singleton source S,
    returns a bound B' and a set of vertices U
    U is a set of size =< k obtained by running 'mini Dijkstra' restricted by B
    and B' is a possibly tighter bound for U
    """
    # --- 1) 입력 검증 & 초기화 -----------------------------
    if not isinstance(S, set) or len(S) != 1:
        print(f"BaseCase called with invalid S={S}")
        raise ValueError("S must be a singleton set {source_vertex}")
    x = next(iter(S))        # 소스 꼭짓점
    U_0 = set(S)             # 현재까지 확정된 정점 집합 (최대 k+1개)

    # --- 2) 미니 다이크스트라 -----------------------------
    heap = [(dist[x], x)]    # (거리, 정점)
    heapq.heapify(heap)

    while heap and len(U_0) < k + 1:
        d_u, u = heapq.heappop(heap)

        # 낡은 키/Bound 체크
        if d_u != dist[u] or d_u >= B:
            continue

        if u not in U_0:
            U_0.add(u)

        # dict → (v,w)로 명시적으로 순회 (형식 혼동 방지)
        for v, w in adj[u].items():
            nd = d_u + w

            # 완화가 실제로 일어나면 즉시 갱신 + push
            if nd < dist[v] or (nd == dist[v] and (depth[u] + 1, u) < (depth[v], pred[v])):
                if nd < B:                           # ★ Bound 안에서만
                    dist[v]  = nd
                    depth[v] = depth[u] + 1
                    pred[v]  = u
                    heapq.heappush(heap, (nd, v))
            else:
                # ★ equality-edge를 따라 탐색 전개 (이미 최단거리여도)
                if nd == dist[v] and nd < B:
                    heapq.heappush(heap, (dist[v], v))

    # --- 3) 정점 수(k+1) 에 따라 반환 ----------------------
    if len(U_0) <= k:                 # 아직 k개 이하라면
        return B, U_0                 #  → B는 그대로
    else:                             # k+1개가 모이면
        farthest = max(U_0, key=lambda v: dist[v])   # 가장 먼 정점
        B_p = dist[farthest]          # 더 타이트한 B'
        U_0.remove(farthest)          # U는 k개로 줄여서 반환
        return B_p, U_0

# Algorithm 3
def BMSSP(l, B, S):
    global adj, vertices, dist, depth, pred, k, t
    """
    Given an integer level 0 =< l =< ceiling(logn/t)), source set S of size =< 2^(lt),
    and an upper bound B > max {dist(x) | x in S},
    returns a bound B' =< B and a set of (note: all complete) vertices U
    It is required that for every incomplete x with dist(x) < B, 
    the shortest path to x visits some complete vertex y in S
    """
    if l == 0:
        res = BaseCase(B, S)   # ★ 한 번만!
        print(res)
        return res
    (P, W) = findPivots(B, S)
    M = 2**((l-1)*t)
    print(f"BMSSP: l={l}, B={B}, |S|={len(S)}, |P|={len(P)}, |W|={len(W)}, M={M}")
    D = DataStructureD(M, B)
    for x in P:
        D.insert(x, (dist[x], depth[x], pred[x], x))

    #   B′0 ← min_{x∈P} d̂[x]   (P가 비면 B 자체)
    if not P:
        print("=====P is empty=====")
    if P:
        print(f"=====P is not empty: of size {len(P)}=====")
    B0 = min((dist[x] for x in P), default=B)
    B_last = B0
    U      = set()
    i      = 0

    # Successful execution if D is empty (B' = B)
    # Partial execution if len(U) > k*2**(l*t) (due to large workload, B' < B)
    while len(U) < k*2**(l*t) and D:
        #U_cap = k * 2**(l*t)
        #print(f"[l={l}] guard: |U|={len(U)} < cap={U_cap} and D={bool(D)}")
        i = i+1
        # print(f"D count: {D.nnz}")
        B_i, S_i = D.pull()
        print(f"BMSSP: i={i}, B_i={B_i}, |S_i|={len(S_i)}")
        # 이번 회차 재귀 호출
        B_prime_i, U_i = BMSSP(l - 1, B_i, S_i)
        B_last = B_prime_i        # ★ 마지막 B′ 갱신
        U = U.union(U_i)
        K = []
        for u in U_i:
            for v in adj[u]:
                if needsUpdate(u, v):
                    update(u, v)
                if dist[u] + adj[u][v] >= B_i and dist[u] + adj[u][v] < B:
                    D.insert(v, (dist[u] + adj[u][v], depth[u] + 1, u, v))
                elif dist[u] + adj[u][v] >= B_prime_i and dist[u] + adj[u][v] < B_i:
                    K.append((v, (dist[u] + adj[u][v], depth[u] + 1, u, v)))
        for x in S_i:
            if dist[x] >= B_prime_i and dist[x] < B_i:
                K.append((x, (dist[x], depth[x], pred[x], x)))
        D.batch_prepend(K)
        # BMSSP while 바디 끝부분, D.batch_prepend(K) 바로 다음
        #print(f"[l={l}] after prepend: |K|={len(K)}, D_empty={not D}")

    # ----- 3. 종료 처리 ---------------------------------------
    B_final = min(B_last, B)                     # 22 행과 동일
    U.update(x for x in W if dist[x] < B_final)  # U ← U ∪ { … }

    return B_final, U  # B', U 