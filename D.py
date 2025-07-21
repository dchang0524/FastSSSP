import bisect
import heapq

class Node:
    __slots__ = ('key', 'prev', 'next')
    def __init__(self, key):
        self.key = key
        self.prev = None
        self.next = None

class Block:
    def __init__(self, is_prepended=False):
        # Doubly linked list of Node; head insertion
        self.head = None
        self.size = 0
        self.is_prepended = is_prepended
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def insert_front(self, node, arr):
        node.prev = None
        node.next = self.head
        if self.head:
            self.head.prev = node
        self.head = node
        self.size += 1
        v = arr[node.key]
        if v < self.min_val:
            self.min_val = v
        if v > self.max_val:
            self.max_val = v

    def remove_node(self, node):
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if self.head is node:
            self.head = node.next
        self.size -= 1

    def recalc_bounds(self, arr):
        v_min, v_max = float('inf'), float('-inf')
        cur = self.head
        while cur:
            v = arr[cur.key]
            if v < v_min:
                v_min = v
            if v > v_max:
                v_max = v
            cur = cur.next
        self.min_val = v_min if self.size > 0 else float('inf')
        self.max_val = v_max if self.size > 0 else float('-inf')

class DataStructureD:
    def __init__(self, M, B):
        """
        Initialize with block size M and global bound B.
        Keys can be any hashable objects (e.g., tuples).
        """
        self.M = M
        self.B = B
        self.arr = {}             # key -> current value
        self.node_map = {}        # key -> Node
        self.block_map = {}       # key -> Block
        self.D0 = []              # list of prepended Blocks
        self.D1_bounds = []       # sorted list of block.max_val for D1
        self.D1_blocks = []       # parallel list of Blocks

    def _remove_block(self, block):
        if block.is_prepended:
            self.D0.remove(block)
        else:
            idx = self.D1_blocks.index(block)
            self.D1_blocks.pop(idx)
            self.D1_bounds.pop(idx)

    def _find_D1_index(self, b):
        idx = bisect.bisect_left(self.D1_bounds, b)
        return idx if idx < len(self.D1_bounds) else len(self.D1_bounds) - 1

    def insert(self, key, b):
        """
        Insert or decrease-key: set key->b if b < old value.
        """
        old = self.arr.get(key)
        if key in self.node_map:
            if b >= old:
                return
            # remove from old block
            old_block = self.block_map[key]
            node = self.node_map[key]
            old_block.remove_node(node)
            if old_block.size == 0:
                self._remove_block(old_block)
            else:
                old_block.recalc_bounds(self.arr)
            del self.node_map[key]
            del self.block_map[key]
        # update value
        self.arr[key] = b
        node = Node(key)
        self.node_map[key] = node
        # place into D1
        if not self.D1_blocks:
            blk = Block(is_prepended=False)
            blk.insert_front(node, self.arr)
            self.D1_blocks.append(blk)
            self.D1_bounds.append(blk.max_val)
            self.block_map[key] = blk
        else:
            idx = self._find_D1_index(b)
            blk = self.D1_blocks[idx]
            blk.insert_front(node, self.arr)
            blk.recalc_bounds(self.arr)
            self.D1_bounds[idx] = blk.max_val
            self.block_map[key] = blk
            if blk.size > self.M:
                self._split_block(idx)

    def _partition_keys(self, keys, pivot_val):
        lows, pivots, highs = [], [], []
        for k in keys:
            v = self.arr[k]
            if v < pivot_val:
                lows.append(k)
            elif v > pivot_val:
                highs.append(k)
            else:
                pivots.append(k)
        return lows, pivots, highs

    def _select(self, keys, k):
        n = len(keys)
        if n <= 10:
            return sorted(keys, key=lambda x: self.arr[x])[k]
        # groups of 5
        groups = [keys[i:i+5] for i in range(0, n, 5)]
        medians = [sorted(g, key=lambda x: self.arr[x])[len(g)//2] for g in groups]
        pivot = self._select(medians, len(medians)//2)
        pv = self.arr[pivot]
        lows, pivots, highs = self._partition_keys(keys, pv)
        if k < len(lows):
            return self._select(lows, k)
        if k < len(lows) + len(pivots):
            return pivots[0]
        return self._select(highs, k - len(lows) - len(pivots))

    def _split_block(self, idx):
        blk = self.D1_blocks[idx]
        # collect keys
        keys = []
        cur = blk.head
        while cur:
            keys.append(cur.key)
            cur = cur.next
        m = len(keys) // 2
        med_key = self._select(keys, m)
        pv = self.arr[med_key]
        lows, pivots, highs = self._partition_keys(keys, pv)
        # fix sizes
        if len(lows) > m:
            extra = len(lows) - m
            for k in list(lows):
                if extra == 0:
                    break
                if self.arr[k] == pv:
                    lows.remove(k)
                    highs.insert(0, k)
                    extra -= 1
        elif len(lows) < m:
            need = m - len(lows)
            lows += pivots[:need]
            highs = pivots[need:] + highs
        # build low block
        low_block = Block(is_prepended=False)
        for k in lows:
            node = self.node_map[k]
            node.prev = node.next = None
            low_block.insert_front(node, self.arr)
            self.block_map[k] = low_block
        # build high block
        high_block = Block(is_prepended=False)
        for k in highs:
            node = self.node_map[k]
            node.prev = node.next = None
            high_block.insert_front(node, self.arr)
            self.block_map[k] = high_block
        # replace in lists
        self.D1_blocks[idx] = low_block
        self.D1_bounds[idx] = low_block.max_val
        self.D1_blocks.insert(idx+1, high_block)
        self.D1_bounds.insert(idx+1, high_block.max_val)

    def _current_min(self):
        candidates = [blk.min_val for blk in self.D0 + self.D1_blocks]
        return min(candidates) if candidates else self.B

    def _chunk_keys(self, keys):
        if len(keys) <= self.M:
            return [keys]
        pivot = self._select(keys, len(keys)//2)
        pv = self.arr[pivot]
        lows, pivots, highs = self._partition_keys(keys, pv)
        left = lows + pivots
        chunks = []
        chunks.extend(self._chunk_keys(left))
        if highs:
            chunks.extend(self._chunk_keys(highs))
        return chunks

    def batch_prepend(self, keys):
        if not keys:
            return
        # must all be <= current min
        if any(self.arr[k] > self._current_min() for k in keys):
            return  # discard entire batch
        # sort by value then chunk
        keys.sort(key=lambda k: self.arr[k])
        for chunk in self._chunk_keys(keys):
            blk = Block(is_prepended=True)
            for k in chunk:
                # remove old if exists
                if k in self.node_map:
                    ob = self.block_map[k]
                    ob.remove_node(self.node_map[k])
                    if ob.size == 0:
                        self._remove_block(ob)
                node = Node(k)
                self.node_map[k] = node
                self.arr[k] = self.arr.get(k)
                blk.insert_front(node, self.arr)
                self.block_map[k] = blk
            self.D0.insert(0, blk)

    def pull(self):
        # select blocks until >= M
        selected = []
        total = 0
        for blk in list(self.D0):
            selected.append(blk)
            total += blk.size
            if total >= self.M:
                break
        for blk in list(self.D1_blocks):
            if total >= self.M:
                break
            selected.append(blk)
            total += blk.size
        # prepare sorted lists per block
        lists = []
        for blk in selected:
            keys = []
            cur = blk.head
            while cur:
                keys.append(cur.key)
                cur = cur.next
            lists.append(sorted(keys, key=lambda k: self.arr[k]))
        merged = heapq.merge(*lists, key=lambda k: self.arr[k])
        S = []
        for _ in range(self.M):
            try:
                S.append(next(merged))
            except StopIteration:
                break
        # remove S from blocks
        for blk in selected[:]:
            count_in_blk = sum(1 for k in S if self.block_map.get(k) is blk)
            if count_in_blk >= blk.size:
                # remove entire block
                cur = blk.head
                while cur:
                    k = cur.key
                    del self.node_map[k]; del self.block_map[k]
                    cur = cur.next
                self._remove_block(blk)
            else:
                # partial
                for k in S:
                    if self.block_map.get(k) is blk:
                        node = self.node_map[k]
                        blk.remove_node(node)
                        del self.node_map[k]; del self.block_map[k]
                blk.recalc_bounds(self.arr)
                if not blk.is_prepended:
                    idx = self.D1_blocks.index(blk)
                    self.D1_bounds[idx] = blk.max_val
        # compute next separator
        mins = [blk.min_val for blk in self.D0 + self.D1_blocks]
        x = min(mins) if mins else self.B
        return S, x
