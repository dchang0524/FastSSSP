import math
from math import log2
import heapq

N = 100000
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
    new_vertices = [[] for _ in range(N)]
    new_adj = {}

    # create new vertices (neighbor pairs)
    for i in range(N):
        for x in adj[i]:
            new_vertices[i].append((i, x[0]))
            new_vertices[x[0]].append((x[0], i))
    
    for v_pairs in new_vertices:
        for x in v_pairs:
            new_adj[x] = set()
    
    # make a cycle per original vertex
    i = 0
    while i < N:
        j = 0
        while j < len(new_vertices[i]):
            if j < len(new_vertices[i]) - 1:
                new_adj[new_vertices[i][j]].add((new_vertices[i][j+1], 0))
            else:
                new_adj[new_vertices[i][j]].add((new_vertices[i][0], 0))
            j = j+1
        i = i+1
    
    # add edges between vertices with weights same as original
    for key in adj:
        for x in adj[key]:
            new_adj[(key, x[0])].add(((x[0], key), x[1]))
    N = sum(len(sublist) for sublist in new_vertices)
    adj = new_adj
    start = (start, next(iter(adj[start])))
    vertices = new_vertices
    return new_adj
          

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
    if dist[u] != dist[v]:
        return (dist[v]-dist[u])/abs(dist[v]-dist[u])
    if depth[u] != depth[v]:
        return (depth[v]-depth[u])/abs(depth[v]-depth[u])
    return (v-u)/abs(v-u)

#Check if edge (u,v) needs to be relaxed(added)
def needsUpdate(u, v):
    if dist[u] == -1:
        return 0
    weight = adj[u][v]
    if (weight + dist[u] != dist[v]):
        cDist = weight + dist[u]
        return (dist[v] - cDist)/abs(dist[v] - cDist) 
    if (depth[u] + 1 != depth[v]):
        return (depth[v] - depth[u] - 1)/abs(depth[v] - depth[u] - 1)
    if pred[v] == u:
        return 1
    return cmp(u, pred[v])

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
    D.Initialize(M, B)
    for x in P:
        D.Insert((dist[x], x))

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
                    D.Insert((dist[u] + adj[u][v], v))
                elif dist[u] + adj[u][v] >= B_ip and dist[u] + adj[u][v] < B_i:
                    K.add((dist[u] + adj[u][v], v))
        for x in S_i:
            if dist[x] >= B_ip and dist[x] < B_i:
                K.add((dist[x], x))
        D.BatchPrepend(K)

    B_p.append(B)
    B_p = min(B_p)
    for x in W:
        if dist[x] < B_p:
            U.add(x)
    return B_p, U  # B', U      