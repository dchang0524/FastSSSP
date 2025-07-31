from sortedcontainers import SortedDict
import bisect

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class Block:
    def __init__(self, bound, is_prepended=False):
        self.head = None
        self.size = 0
        self.bound = bound            # TreeMap key for this block
        self.is_prepended = is_prepended

    def insert_front(self, node):
        node.next = self.head
        node.prev = None
        if self.head:
            self.head.prev = node
        self.head = node
        self.size += 1
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
        self.node_map = {}          # key -> Node
        self.block_map = {}         # key -> Block
        self.D0 = []                # list of prepended Blocks
        self.D1 = SortedDict()      # TreeMap: bound -> Block
        self.global_min = B

    def _remove_block(self, blk):
        if blk.is_prepended:
            self.D0.remove(blk)
        else:
            # direct delete via block.bound
            del self.D1[blk.bound]

    def delete(self, key):
        node = self.node_map.pop(key, None)
        if not node:
            return
        blk = self.block_map.pop(key)
        blk.remove_node(node)
        if blk.size == 0:
            self._remove_block(blk)

    def insert(self, key, value):
        old = self.node_map.get(key)
        if old:
            if value >= old.value:
                return
            self.delete(key)
        node = Node(key, value)
        self.node_map[key] = node
        if value < self.global_min:
            self.global_min = value
        if not self.D1:
            # first block uses B as initial bound
            blk = Block(self.B)
            self.D1[self.B] = blk
        # locate TreeMap entry >= value
        idx = self.D1.bisect_left(value)
        if idx == len(self.D1):
            idx -= 1
        bound = self.D1.iloc[idx]
        blk = self.D1[bound]
        blk.insert_front(node)
        self.block_map[key] = blk
        if blk.size > self.M:
            self._split_block(bound)

    def _select_median(self, items):
        if len(items) <= 5:
            return sorted(items, key=lambda x: x[1])[len(items)//2]
        groups = [items[i:i+5] for i in range(0, len(items), 5)]
        medians = [sorted(g, key=lambda x: x[1])[len(g)//2] for g in groups]
        return self._select_median(medians)

    def _split_block(self, old_bound):
        blk = self.D1.pop(old_bound)
        items = []
        cur = blk.head
        while cur:
            items.append((cur.key, cur.value))
            cur = cur.next
        med_key, med_val = self._select_median(items)
        low_items = [(k, v) for k, v in items if v <= med_val]
        high_items = [(k, v) for k, v in items if v > med_val]
        low_blk = Block(med_val)
        for k, v in low_items:
            node = Node(k, v)
            low_blk.insert_front(node)
            self.node_map[k] = node
            self.block_map[k] = low_blk
        high_blk = Block(self.B)
        for k, v in high_items:
            node = Node(k, v)
            high_blk.insert_front(node)
            self.node_map[k] = node
            self.block_map[k] = high_blk
        # reinsert both
        self.D1[low_blk.bound] = low_blk
        self.D1[high_blk.bound] = high_blk

    def batch_prepend(self, items):
        # items: list of (key,value) pairs
        if not items:
            return
        # keep only those with value <= current global_min and deduplicate by key (keep smallest v)
        filtered = {}
        for k, v in items:
            if v <= self.global_min:
                if k not in filtered or v < filtered[k]:
                    filtered[k] = v
        items = [(k, filtered[k]) for k in filtered]
        if not items:
            return
        # recursive chunking until each sublist size <= M
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
        """
        Pull up to M smallest unique keys using quickselect and block-tracking.
        """
        # 1) Collect blocks from D0 until at least M items
        blocks0 = []
        count0 = 0
        for blk in self.D0:
            blocks0.append(blk)
            count0 += blk.size
            if count0 >= self.M:
                break
        # 2) Collect blocks from D1 until combined >= M
        blocks1 = []
        count1 = 0
        for blk in self.D1.values():
            if count0 + count1 >= self.M:
                break
            blocks1.append(blk)
            count1 += blk.size
        # 3) Flatten into list of (key, value, block)
        items = []
        for blk in blocks0 + blocks1:
            cur = blk.head
            while cur:
                items.append((cur.key, cur.value, blk))
                cur = cur.next
        # 4) Quickselect threshold
        def quickselect(lst, k):
            if len(lst) <= 1:
                return lst[0][1] if lst else None
            pivot = lst[len(lst)//2][1]
            lows = [x for x in lst if x[1] < pivot]
            highs = [x for x in lst if x[1] > pivot]
            pivs = [x for x in lst if x[1] == pivot]
            if k < len(lows):
                return quickselect(lows, k)
            if k < len(lows) + len(pivs):
                return pivot
            return quickselect(highs, k - len(lows) - len(pivs))
        if len(items) <= self.M:
            threshold = self.B
        else:
            threshold = quickselect(items, self.M - 1)
        # 5) Gather up to M keys in block order, track last blocks
        result = []
        seen = set()
        last0 = None
        last1 = None
        for blk in blocks0 + blocks1:
            cur = blk.head
            while cur and len(result) < self.M:
                k, v = cur.key, cur.value
                if k not in seen and v <= threshold:
                    seen.add(k)
                    result.append(k)
                    if blk.is_prepended:
                        last0 = blk
                    else:
                        last1 = blk
                cur = cur.next
        # 6) Remove full and partial blocks in D0
        if last0:
            idx0 = blocks0.index(last0)
            # remove all earlier blocks fully
            for blk in blocks0[:idx0]:
                cur = blk.head
                while cur:
                    kk = cur.key
                    del self.node_map[kk]
                    del self.block_map[kk]
                    cur = cur.next
                self.D0.remove(blk)
            # partial removal in last0
            for k in [k for k in result if self.block_map.get(k) is last0]:
                node = self.node_map[k]
                last0.remove_node(node)
                del self.node_map[k]
                del self.block_map[k]
        # 7) Remove full and partial blocks in D1
        if last1:
            idx1 = blocks1.index(last1)
            # remove all earlier blocks fully
            for blk in blocks1[:idx1]:
                cur = blk.head
                while cur:
                    kk = cur.key
                    del self.node_map[kk]
                    del self.block_map[kk]
                    cur = cur.next
                del self.D1[blk.bound]
            # partial removal in last1
            for k in [k for k in result if self.block_map.get(k) is last1]:
                node = self.node_map[k]
                last1.remove_node(node)
                del self.node_map[k]
                del self.block_map[k]
        # 8) Compute separator x by inspecting the head of the first blocks in D0 and D1
        vals = []
        # first prepended block
        if self.D0 and self.D0[0].head:
            vals.append(self.D0[0].head.value)
        # first regular block
        if self.D1:
            first_bound = self.D1.keys()[0]
            first_blk = self.D1[first_bound]
            if first_blk.head:
                vals.append(first_blk.head.value)
        x = min(vals) if vals else self.B
        return result, x
