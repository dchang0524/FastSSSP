from sortedcontainers import SortedDict
# No longer need to import algorithms, which solves the circular import
# import algorithms as alg 

class Node:
    def __init__(self, key, value_tuple):
        self.key = key
        # Value is now a tuple, e.g., (distance, depth, key)
        self.value = value_tuple
        self.prev = None
        self.next = None

class Block:
    def __init__(self, bound_tuple, is_prepended=False):
        self.head = None
        self.size = 0
        self.bound = bound_tuple
        self.is_prepended = is_prepended

    def insert_front(self, node):
        node.next = self.head
        node.prev = None
        if self.head:
            self.head.prev = node
        self.head = node
        self.size += 1

    def remove_node(self, node):
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if self.head is node:
            self.head = node.next
        self.size -= 1

class DataStructureD:
    def __init__(self, M: int, B):
        self.M = M # maximum size of a block

        # ① B를 항상 3-튜플로 보존 + 스칼라 값도 따로 저장
        if isinstance(B, tuple):
            self.B_tuple = B
            self.B_scalar = B[0]
        else:
            self.B_tuple = (B, float('inf'), float('inf'))
            self.B_scalar = B

        # B is now an tuple, e.g., (B, float('inf'), float('inf'))
        self.B = self.B_tuple
        self.node_map = {}          # key -> Node
        self.block_map = {}         # key -> Block
        self.D0 = []                # list of prepended Blocks
        self.D1 = SortedDict()      # TreeMap: bound_tuple -> Block
        # global_min is also a tuple now
        self.global_min = self.B
        self.nnz = 0                # ★ 전체 노드 개수

    def _remove_block(self, blk):
        if blk.is_prepended:
            self.D0.remove(blk)
        else:
            del self.D1[blk.bound]

    def delete(self, key):
        node = self.node_map.pop(key, None)
        if not node:
            return
        blk = self.block_map.pop(key)
        blk.remove_node(node)
        self.nnz -= 1               # ★ 카운터 갱신
        if blk.size == 0:
            self._remove_block(blk)

    def insert(self, key, value_tuple):
        old_node = self.node_map.get(key)
        # Python's tuple comparison works lexicographically out of the box
        if old_node and value_tuple >= old_node.value:
            return
        if old_node:
            self.delete(key)
            
        node = Node(key, value_tuple)
        self.node_map[key] = node

        if value_tuple < self.global_min:
            self.global_min = value_tuple
        
        if not self.D1:
            blk = Block(self.B)
            self.D1[self.B] = blk
        
        # bisect_left works correctly on the sorted tuple keys of D1
        idx   = self.D1.bisect_left(value_tuple)     # 위치 탐색
        keys  = list(self.D1.keys())                 # key-view → list

        # idx 가 len(keys) 인 경우 → 마지막 블록을 선택
        if idx == len(keys):
            bound_tuple = keys[-1]
        else:
            bound_tuple = keys[idx]

        blk = self.D1[bound_tuple]
        
        blk.insert_front(node)
        self.block_map[key] = blk

        self.nnz += 1               # ★ 카운터 갱신
        
        if blk.size > self.M:
            self._split_block(bound_tuple)

    def _select_median(self, items):
        # items is a list of [(key, value_tuple)]
        # The default sort key (the whole item) is fine, but sorting by the value_tuple is more explicit.
        if len(items) <= 5:
            return sorted(items, key=lambda x: x[1])[len(items)//2]
        
        groups = [items[i:i+5] for i in range(0, len(items), 5)]
        medians = [sorted(g, key=lambda x: x[1])[len(g)//2] for g in groups]
        return self._select_median(medians)

    def quickselect(self, lst, k):
        if not lst:
            return None
        
        # The median is a tuple: (key, value_tuple)
        median_item = self._select_median(lst)
        # The pivot is the value_tuple itself
        pivot = median_item[1]

        lows = [x for x in lst if x[1] < pivot]
        highs = [x for x in lst if x[1] > pivot]
        pivs = [x for x in lst if x[1] == pivot]

        if k < len(lows):
            return self.quickselect(lows, k)
        elif k < len(lows) + len(pivs):
            return pivot # Return the pivot tuple
        else:
            return self.quickselect(highs, k - len(lows) - len(pivs))

    def _split_block(self, old_bound_tuple):
        blk = self.D1.pop(old_bound_tuple)
        items = []
        cur = blk.head
        while cur:
            items.append((cur.key, cur.value))
            cur = cur.next

        # median_item is (key, value_tuple)
        median_item = self._select_median(items)
        # med_val is the value_tuple
        med_val_tuple = median_item[1]

        low_items = [(k, v) for k, v in items if v <= med_val_tuple]
        high_items = [(k, v) for k, v in items if v > med_val_tuple]
        
        low_blk = Block(med_val_tuple)
        for k, v_tuple in low_items:
            node = Node(k, v_tuple)
            low_blk.insert_front(node)
            self.block_map[k] = low_blk
        
        high_blk = Block(old_bound_tuple)
        for k, v_tuple in high_items:
            node = Node(k, v_tuple)
            high_blk.insert_front(node)
            self.block_map[k] = high_blk
        
        self.D1[low_blk.bound] = low_blk
        # Only add high_blk if it's not empty
        if high_blk.size > 0:
            self.D1[high_blk.bound] = high_blk

    def batch_prepend(self, items):
        # items: list of (key, value_tuple) pairs
        if not items:
            return
        
        filtered = {}
        for k, v_tuple in items:
            if v_tuple < self.global_min:
                if k not in filtered or v_tuple < filtered[k]:
                    filtered[k] = v_tuple
        
        items = list(filtered.items())
        if not items:
            return
        
        def chunk(lst):
            if len(lst) <= self.M:
                return [lst]
            # _select_median expects [(key, value)], so we pass the list directly
            med_item = self._select_median(lst)
            med_tuple = med_item[1]
            left = [x for x in lst if x[1] <= med_tuple]
            right = [x for x in lst if x[1] > med_tuple and x != med_item]
            return chunk(left) + (chunk(right) if right else [])

        for group in reversed(chunk(items)):
            # The bound doesn't matter as much for D0 blocks, but should be a tuple
            blk = Block(self.B, is_prepended=True)
            for k, v_tuple in group:
                if k in self.node_map:
                    self.delete(k)          # 중복이면 nnz-1 처리
                node = Node(k, v_tuple)
                blk.insert_front(node)

                self.nnz += 1                     # ★ 한 노드 삽입 → nnz +1
                
                self.node_map[k] = node
                self.block_map[k] = blk
            self.D0.insert(0, blk)

    def pull(self):
        """
        Return (x, S) where
        • S : set of ≤ M keys whose (distance, depth, pred) tuples are the smallest in D
        • x : scalar lower-bound that separates S from the remaining elements
        After the call every key in S is physically removed from the data structure.
        If the structure becomes empty we return (B, ∅).
        """
        # 1)  ── Collect a prefix of blocks until ≥ M items ─────────────────────────
        collected = []
        count = 0

        for blk in self.D0:                      # first scan prepended blocks
            collected.append(blk)
            count += blk.size
            if count >= self.M:
                break

        if count < self.M:                       # then scan inserted blocks
            for _, blk in self.D1.items():
                collected.append(blk)
                count += blk.size
                if count >= self.M:
                    break

        # 2)  ── Flatten the nodes inside the collected blocks ─────────────────────
        items = []                               # [(key, value_tuple)]
        for blk in collected:
            n = blk.head
            while n:
                items.append((n.key, n.value))
                n = n.next

        if not items:                            # nothing left in the structure
            return self.B[0] if isinstance(self.B, tuple) else self.B, set()

        # 3)  ── Select at most M smallest items  (ties ⇒ arbitrary truncation) ────
        if len(items) > self.M:
            # find the M-th smallest value-tuple via quick-select (O(M))
            threshold = self.quickselect(items, self.M - 1)

            # strictly smaller values
            smaller = [kv for kv in items if kv[1] < threshold]

            # ties that are == threshold
            equal   = [kv for kv in items if kv[1] == threshold]

            needed  = self.M - len(smaller)
            equal   = equal[:needed]            # cut ties so that |S| ≤ M

            chosen  = smaller + equal
        else:
            chosen  = items                     # already ≤ M

        S_keys = {kv[0] for kv in chosen}       # set of keys to return

        # 4)  ── Delete the chosen keys (block-aware fast removal) ─────────────────
        for key in S_keys:
            self.delete(key)                    # O(1) amortised per deletion

        # 5)  ── Compute new separator x  (= smallest remaining value) ─────────────
        remaining = []
        if self.D0 and self.D0[0].head:
            remaining.append(self.D0[0].head.value)
        if self.D1:
            first_blk = self.D1[self.D1.keys()[0]]
            if first_blk.head:
                remaining.append(first_blk.head.value)

        x_tuple  = min(remaining) if remaining else self.B
        x_scalar = x_tuple[0] if isinstance(x_tuple, tuple) else x_tuple

        # 6)  ── Return (bound, keys) in the order expected by BMSSP ───────────────
        return x_scalar, S_keys
    

    # Truthiness 정의
    def __bool__(self):
        return self.nnz > 0         # ★ 선형 순회 없음
        # D0: deque of blocks
        for blk in self.D0:
            if blk.size > 0:
                return True
        # D1: SortedDict(bound → block)
        for blk in self.D1.values():
            if blk.size > 0:
                return True
        return False
    
    ###########################
    # __bool__() is needed for 'if D' check in the main algorithm.
    ###########################

