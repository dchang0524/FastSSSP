import random

N = 100000

adj = []
existing_edges = set() # Use a set for O(1) average time lookups

# --- Step 1: Generate a spanning tree (O(N) time) ---
# print("Generating spanning tree...")
for i in range(1, N):
    j = random.randrange(0, i)
    weight = random.randint(1, 20)
    
    # Add edge to both representations
    adj.append((j, i, weight))
    existing_edges.add((j, i))

# --- Step 2: Add N-1 extra edges (O(N) time) ---
# print("Adding extra edges...")
count = 0
while count < N-1:
    u = random.randrange(0, N)
    v = random.randrange(0, N)
    
    # Ensure u != v and we don't create parallel edges
    if u == v:
        continue
    
    # To make the check canonical, always store the smaller index first
    edge = tuple(sorted((u, v)))
    if edge in existing_edges:
        continue
        
    weight = random.randint(1, 10)
    adj.append((edge[1], edge[0], weight))
    existing_edges.add(edge)
    count += 1

#print("Graph is generated.")
# print(adj[N-100:N+100])