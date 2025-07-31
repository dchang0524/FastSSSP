import bisect

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class Block:
    def __init__(self, bound, is_prepended=False):
        # Doubly-linked list head, block bounds, and flags
        self.head = None
        self.size = 0
        self.bound = bound            # upper bound (TreeMap key)
        self.is_prepended = is_prepended
        self.min_val = float('inf')   # track min for separator

    def insert_front(self, node):
        # Insert node at front of linked list
        node.next = self.head
        node.prev = None
        if self.head:
            self.head.prev = node
        self.head = node
        self.size += 1
        # update bounds
        self.min_val = min(self.min_val, node.value)
        self.bound = max(self.bound, node.value)

    def remove_node(self, node):
        # Unlink node
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if self.head is node:
            self.head = node.next
        self.size -= 1

class DataStructureD:
    def __init__(self, M, B):
        self.M = M
        self.B = B
        self.node_map = {}       # key -> Node
        self.block_map = {}      # key -> Block
        self.D0 = []             # list of prepended Blocks
        # D1 as a TreeMap: bound -> Block, with sorted list of keys
        self.D1 = {}
        self.D1_keys = []       # sorted bounds
        self.global_min = B      # min value across all nodes

    def _remove_block(self, blk):
        # Remove empty block from D0 or D1
        if blk.is_prepended:
            self.D0.remove(blk)
        else:
            bound = blk.bound
            del self.D1[bound]
            self.D1_keys.remove(bound)

    def delete(self, key):
        # Remove key from its block
        node = self.node_map.pop(key, None)
        if not node:
            return
        blk = self.block_map.pop(key)
        blk.remove_node(node)
        if blk.size == 0:
            self._remove_block(blk)

    def insert(self, key, value):
        # Insert or decrease-key
        old_node = self.node_map.get(key)
        if old_node:
            if value >= old_node.value:
                return
            self.delete(key)
        # create new node
        node = Node(key, value)
        self.node_map[key] = node
        # update global min
        if value < self.global_min:
            self.global_min = value
        # ensure at least one D1 block
        if not self.D1_keys:
            blk = Block(self.B)
            self.D1[self.B] = blk
            self.D1_keys.append(self.B)
        # find TreeMap entry >= value
        idx = bisect.bisect_left(self.D1_keys, value)
        if idx == len(self.D1_keys):
            idx -= 1
        bound = self.D1_keys[idx]
        blk = self.D1[bound]
        blk.insert_front(node)
        self.block_map[key] = blk
        # if block too large, split
        if blk.size > self.M:
            self._split_block(bound)

    def _select_median(self, items):
        if len(items) <= 5:
            return sorted(items, key=lambda x: x[1])[len(items)//2]
        groups = [items[i:i+5] for i in range(0, len(items), 5)]
        medians = [sorted(g, key=lambda x: x[1])[len(g)//2] for g in groups]
        return self._select_median(medians)

    def _split_block(self, old_bound):
        # Split block at old_bound into two around median
        blk = self.D1.pop(old_bound)
        self.D1_keys.remove(old_bound)
        # collect items
        items = []
        cur = blk.head
        while cur:
            items.append((cur.key, cur.value))
            cur = cur.next
        med_key, med_val = self._select_median(items)
        # partition
        low_items = [(k, v) for (k, v) in items if v <= med_val]
        high_items = [(k, v) for (k, v) in items if v > med_val]
        # build low block
        low_blk = Block(med_val)
        for k, v in low_items:
            node = Node(k, v)
            low_blk.insert_front(node)
            self.node_map[k] = node
            self.block_map[k] = low_blk
        # build high block
        high_blk = Block(self.B)
        for k, v in high_items:
            node = Node(k, v)
            high_blk.insert_front(node)
            self.node_map[k] = node
            self.block_map[k] = high_blk
        # reinsert into D1
        self.D1[low_blk.bound] = low_blk
        self.D1[high_blk.bound] = high_blk
        bisect.insort(self.D1_keys, low_blk.bound)
        bisect.insort(self.D1_keys, high_blk.bound)

    def batch_prepend(self, items):
        # items: list of (key,value) pairs
        if not items:
            return
        # discard batch if any value > global_min
        if any(v > self.global_min for _, v in items):
            return
        # recursive chunking
        def chunk(lst):
            if len(lst) <= self.M:
                return [lst]
            med = self._select_median(lst)
            left = [x for x in lst if x[1] <= med[1]]
            right = [x for x in lst if x[1] > med[1]]
            return chunk(left) + (chunk(right) if right else [])
        for group in chunk(items):
            blk = Block(self.B, is_prepended=True)
            for k, v in group:
                if k in self.node_map:
                    self.delete(k)
                node = Node(k, v)
                blk.insert_front(node)
                self.node_map[k] = node
                self.block_map[k] = blk
            self.D0.insert(0, blk)

    def pull(self):
        # collect blocks until >= M
        selected = []
        total = 0
        for blk in list(self.D0):
            selected.append(blk)
            total += blk.size
            if total >= self.M:
                break
        for bound in self.D1_keys:
            if total >= self.M:
                break
            blk = self.D1[bound]
            selected.append(blk)
            total += blk.size
        # flatten & sort
        items = []
        for blk in selected:
            cur = blk.head
            while cur:
                items.append((cur.key, cur.value))
                cur = cur.next
        items.sort(key=lambda x: x[1])
        result = [k for k,_ in items[:self.M]]
        # remove pulled
        for blk in selected[:]:
            cnt = sum(1 for k in result if self.block_map.get(k) is blk)
            if cnt >= blk.size:
                # remove whole
                cur = blk.head
                while cur:
                    k = cur.key
                    del self.node_map[k]
                    del self.block_map[k]
                    cur = cur.next
                if blk.is_prepended:
                    self.D0.remove(blk)
                else:
                    del self.D1[blk.bound]
                    self.D1_keys.remove(blk.bound)
            else:
                # partial remove
                for k in result:
                    if self.block_map.get(k) is blk:
                        node = self.node_map[k]
                        blk.remove_node(node)
                        del self.node_map[k]
                        del self.block_map[k]
        # separator
        sep = min((blk.min_val for blk in self.D0 + list(self.D1.values())), default=self.B)
        return result, sep
