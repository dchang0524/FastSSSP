from math import log2

N = 100000
k = log2(N) ** (1/3)
adj = [set() for _ in range(N)]
dist = [-1] * N
len = [-1] * N
pred = [-1] * N




#
def findPivots(B: int, S: set, adj : list):
    W = S.copy()
    prev = S.copy()
    for i in range(1, k+1):
        W_i = set()
        for w in prev:
            for v in adj[w]:
                if v not in S:
                    W_i.add(v)

