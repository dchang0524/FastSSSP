import random

N = 7
tree_weights = random.sample(range(1, 20), N-1)
adj = []

for i in range(N):
    if i == 0:
        continue
    j = random.randrange(0, i)
    adj.append((j, i, tree_weights[i-1]))

print(adj)