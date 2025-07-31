import random

N = 7
tree_weights = random.sample(range(1, 20), N-1)
extra_weights = random.sample(range(1, 10), N-1)
adj = []

for i in range(N):
    if i == 0:
        continue
    j = random.randrange(0, i)
    adj.append((j, i, tree_weights[i-1]))

count = 0
while count < N-1:
    count += 1
    u = random.randrange(1, N)
    v = random.randrange(0, u)
    if (v, u, tree_weights[u-1]) in adj:
        continue
    adj.append((u, v, extra_weights[count-1]))

print(adj)