import math
from math import log2
import heapq
from D import DataStructureD

N = 100000
M = 0
k = math.floor(log2(N) ** (1/3))
t = math.floor(log2(N) ** (2/3))
start = 1
vertices = [i for i in range(N)]
adj = [set() for _ in range(N)]
dist = [-1] * N  # sum of path weights
depth = [-1] * N   # number of vertices traversed
pred = [-1] * N  #last previous vertex visited in path

#Graph Transformation
def transformGraph(graph: dict):
    global N, M, start, adj, vertices, dist, depth, pred

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

    # Re-initialize path-finding arrays for the new graph size
    dist = [-1] * N
    depth = [-1] * N
    pred = [-1] * N

    ###########################
    # Do we have to recompute k and t?
    ###########################
          

#Path Comparison
def cmp(u, v):
    """
    Compares path ending at u, v
    Returns 1 if the path ending at u is shorter, -1 otherwise
    Returns 0 if an error occured (i.e there is no path ending at u/v) 
    Note: we use lexical order to break ties
    """
    if dist[u] == -1 or dist[v] == -1:
        return 0
    if u == v:
        return 0
    if dist[u] != dist[v]:
        return (dist[v]-dist[u])//abs(dist[v]-dist[u])
    if depth[u] != depth[v]:
        return (depth[v]-depth[u])//abs(depth[v]-depth[u])
    return (v-u)//abs(v-u)

#Check if edge (u,v) needs to be relaxed(added). Returns 1 if it does, -1 otherwise
def needsUpdate(u, v):
    if dist[u] == -1:
        return 0
    weight = adj[u][v]
    if (weight + dist[u] != dist[v]):
        cDist = weight + dist[u]
        return (dist[v] - cDist)//abs(dist[v] - cDist) 
    if (depth[u] + 1 != depth[v]):
        return (depth[v] - depth[u] - 1)//abs(depth[v] - depth[u] - 1)
    if pred[v] == u:
        return 1
    return cmp(u, pred[v])

#Relaxes edge (u,v)
def update(u, v):
    cWeight = adj[u][v]
    dist[v] = dist[u] + cWeight
    depth[v] = depth[u] + 1
    pred[v] = u


#Algorithm 1
def findPivots(B, S):
    """
    U': Contains every incomplete vertex with d(v) < B where the shortest path visits some vertex in S
    Finds a set W of size O(k|S|) and a set of pivots P (subset of S) with |P| <= |W|/k
    For every x in U'
        x is in W and complete (dist[x] = d(x))
        or
        The shortest path to x visits some complete vertex in P
    Runs in O(k|W|) = O(min{k^2|S|, k|U'|})
    """

    W = S.copy()
    prev = S.copy()
    for i in range(1, k+1):
        W_i = set()
        for u in prev:
            for v in adj[u]:
                if needsUpdate(u, v):
                    update(u, v)
                    if dist[v] < B:
                        W_i.add(v)                
        W.add(W_i)
        prev = W_i
        if (len(W) > k*len(S)):
            P = S
            return (P, W)
    #Create Temproary subgraph of currently processed shortest paths (Directed Forest)
    tAdj = {}
    for u in W:
        for v in adj[u]:
            if v in W and pred[v] == u:
                tAdj[u].append[{v: adj[u][v]}]
    #Get elements in S whose subtree has at least K vertices
    dfs = []
    stack = [start]
    while len(stack) > 0:
        curr = stack.pop()
        for v in tAdj[curr]:
            if pred[v] == u:
                stack.append(v)
    stack.reverse()
    sz = {}
    for u in W:
        sz[u] = 1
    for u in dfs:
        for v in tAdj[u]:
            if (pred[v] == u):
                sz[u] += sz[v]
    P = {}
    for u in W:
        if sz[u] >= k:
            P.add(u)
    return (P, W)


# Algorithm 2
def BaseCase(B, S):
    """
    Given a bound B and a singleton source S,
    returns a bound B' and a set of vertices U
    U is a set of size =< k obtained by running 'mini Dijkstra' restricted by B
    and B' is a possibly tighter bound for U
    """
    U_0 = S
    x = S[0]
    heap = [(dist[x], x)]  # Sort by distance
    heapq.heapify(heap)
    while heap and len(U_0) < k+1:
        element = heapq.heappop(heap)
        u = element[1]
        U_0.add(u)
        for v in adj[u]:
            if needsUpdate(u, v) and dist[u] + adj[u][v] < B:
                update(u, v)
                heapq.heappush(heap, (dist[v], v))
    if len(U_0) < k+1:
        return B, U_0
    else:
        B_p = 0
        index = 0
        for i in range(len(U_0)):
            if dist[U_0[i]] > B_p:
                B_p = dist[U_0[i]]
                index = i
        U_0.pop(index)
        U = U_0
        return B_p, U


# Algorithm 3
def BMSSP(l, B, S):
    """
    Given an integer level 0 =< l =< ceiling(logn/t)), source set S of size =< 2^(lt),
    and an upper bound B > max {dist(x) | x in S},
    returns a bound B' =< B and a set of (note: all complete) vertices U
    It is required that for every incomplete x with dist(x) < B, 
    the shortest path to x visits some complete vertex y in S
    """
    if l == 0:
        return BaseCase(B, S)  # B', U
    (P, W) = findPivots(B, S)
    M = 2**((l-1)*t)
    D = DataStructureD(M, B)
    for x in P:
        D.insert(x, dist[x])

    B_p = [0]
    i, B_p[0], U = 0, min(dist[x] for x in P) if P else B, set()

    # Successful execution if D is empty (B' = B)
    # Partial execution if len(U) > k*2**(l*t) (due to large workload, B' < B)
    while len(U) < k*2**(l*t) and D:
        i = i+1
        B_i, S_i = D.Pull()
        B_p.append(0)
        B_p[i], U_i = BMSSP(l-1, B_i, S_i)
        B_ip = B_p[i]
        U = U.union(U_i)
        K = set()
        for u in U_i:
            for v in adj[u]:
                if needsUpdate(u, v):
                    update(u, v)
                if dist[u] + adj[u][v] >= B_i and dist[u] + adj[u][v] < B:
                    D.insert(v, dist[u] + adj[u][v])
                elif dist[u] + adj[u][v] >= B_ip and dist[u] + adj[u][v] < B_i:
                    K.add((v, dist[u] + adj[u][v]))
        for x in S_i:
            if dist[x] >= B_ip and dist[x] < B_i:
                K.add((x, dist[x]))
        D.batch_prepend(K)

    B_p.append(B)
    B_p = min(B_p)
    for x in W:
        if dist[x] < B_p:
            U.add(x)
    return B_p, U  # B', U      