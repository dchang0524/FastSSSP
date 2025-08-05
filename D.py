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
        # The bound is now a tuple
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
    def __init__(self, M, B_tuple):
        self.M = M # maximum size of a block
        # B is now an "infinity" tuple, e.g., (float('inf'), float('inf'), float('inf'))
        self.B = B_tuple
        self.node_map = {}          # key -> Node
        self.block_map = {}         # key -> Block
        self.D0 = []                # list of prepended Blocks
        self.D1 = SortedDict()      # TreeMap: bound_tuple -> Block
        # global_min is also a tuple now
        self.global_min = self.B

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
        idx = self.D1.bisect_left(value_tuple)
        bound_tuple = self.D1.iloc[idx]
        blk = self.D1[bound_tuple]
        
        blk.insert_front(node)
        self.block_map[key] = blk
        
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
                    self.delete(k)
                node = Node(k, v_tuple)
                blk.insert_front(node)
                self.node_map[k] = node
                self.block_map[k] = blk
            self.D0.insert(0, blk)

    def pull(self):
        # 1. Collect a prefix of blocks from D0 and D1 until we have at least M items.
        collected_blocks = []
        item_count = 0
        
        # From prepended blocks (D0)
        for blk in self.D0:
            collected_blocks.append(blk)
            item_count += blk.size
            if item_count >= self.M:
                break
                
        # From regular blocks (D1) if needed
        if item_count < self.M:
            for _, blk in self.D1.items():
                collected_blocks.append(blk)
                item_count += blk.size
                if item_count >= self.M:
                    break

        # 2. Flatten the items from ONLY the collected blocks.
        items = []
        for blk in collected_blocks:
            cur = blk.head
            while cur:
                items.append((cur.key, cur.value))
                cur = cur.next

        if not items:
            return [], self.B

        # 3. Find the smallest M items from the collected list.
        if len(items) <= self.M:
            # All collected items are part of the result
            result_keys = [item[0] for item in items]
        else:
            # Use quickselect to find the threshold value for the M smallest items
            threshold_tuple = self.quickselect(items, self.M - 1)
            
            # Collect all candidates with value <= threshold
            candidates = [item for item in items if item[1] <= threshold_tuple]
            
            # Sort candidates to break ties and ensure we take exactly M smallest
            candidates.sort(key=lambda x: x[1])
            result_keys = [item[0] for item in candidates[:self.M]]

        # 4. Delete the pulled keys from the data structure.
        for key in result_keys:
            self.delete(key)

        # 5. Compute the new separator 'x' (the smallest value remaining).
        new_min_vals = []
        # Check the first remaining prepended block
        if self.D0 and self.D0[0].head:
            new_min_vals.append(self.D0[0].head.value)
            
        # Check the first remaining regular block
        if self.D1:
            # Get the first key (bound) from the sorted dictionary
            first_bound = self.D1.keys()[0]
            first_blk = self.D1[first_bound]
            if first_blk.head:
                new_min_vals.append(first_blk.head.value)
                
        x_tuple = min(new_min_vals) if new_min_vals else self.B

        return result_keys, x_tuple