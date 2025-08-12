# lemma33_data_structure.py
# Naive (correctness-first) implementation compatible with algorithms.py expectations.
# API:
#   - insert(key, value_tuple)
#   - batch_prepend(items)               # no extra bound arg (stick to paper-style)
#   - pull() -> (Bi_scalar, Si_set)
#   - __bool__ for emptiness
#
# Notes:
#   * Values are normalized to tuples so Python's lexicographic tuple-compare works.
#   * D0 and D1 are lists of Blocks; D1 kept sorted by block.upper_bound (tuple).
#   * pull() gathers global min M items (naive) and returns remaining min as Bi.

import bisect
import math

INF = math.inf

def _as_tuple(v):
    """Normalize scalar to a 4-tuple so tuple comparison is consistent."""
    if isinstance(v, tuple):
        return v
    return (v, INF, INF, INF)


class Block:
    """Represents a block of key-value pairs, for D0 or D1."""
    def __init__(self, is_d1=False, upper_bound=None):
        self.is_d1 = is_d1
        self.upper_bound = _as_tuple(upper_bound) if upper_bound is not None else _as_tuple(INF)
        self.elements = []  # list[(key, value_tuple)] sorted by value_tuple

    def __len__(self):
        return len(self.elements)


class Lemma33DataStructure:
    """Naive DS for Lemma 3.3 — correctness over performance."""
    def __init__(self, M, B):
        self.M = max(1, int(M))
        self.B = B
        self.nnz = 0  # convenient counter (optional)

        # D0: newest blocks are at the front (index 0)
        self.D0 = []

        # D1: blocks sorted by upper_bound (tuple)
        sentinel = Block(is_d1=True, upper_bound=_as_tuple(B))
        self.D1 = [sentinel]
        self.d1_upper_bounds = [sentinel.upper_bound]  # list of tuple bounds

        # key -> (value_tuple, block)
        self.key_map = {}

    # ---------------- internal helpers ----------------

    def _ensure_d1_nonempty(self, bound=None):
        """If D1 becomes empty, create a sentinel block so bisect/indexing stays safe."""
        if self.D1:
            return
        ub = _as_tuple(self.B if bound is None else bound)
        blk = Block(is_d1=True, upper_bound=ub)
        self.D1 = [blk]
        self.d1_upper_bounds = [blk.upper_bound]

    def _rebuild_d1_order(self):
        """Keep D1 sorted by upper_bound (tuple)."""
        pairs = [(blk.upper_bound, blk) for blk in self.D1]
        pairs.sort(key=lambda x: x[0])
        self.D1 = [blk for _, blk in pairs]
        self.d1_upper_bounds = [blk.upper_bound for blk in self.D1]
        if not self.D1:
            self._ensure_d1_nonempty()

    def _delete_entry(self, key):
        """Remove (key) from whichever block it lives in; avoid index drift by linear scan."""
        info = self.key_map.pop(key, None)
        if not info:
            return
        value, block = info

        # linear search inside the block to avoid stale indices
        idx = None
        for i, (k, v) in enumerate(block.elements):
            if k == key and v == value:
                idx = i
                break
        if idx is not None:
            del block.elements[idx]
            self.nnz -= 1

        # tidy up empty blocks and update bounds/order
        if not block.elements:
            if block in self.D0:
                self.D0.remove(block)
            elif block in self.D1:
                self.D1.remove(block)
                self._rebuild_d1_order()
                self._ensure_d1_nonempty()
        else:
            if block.is_d1:
                block.upper_bound = block.elements[-1][1]
                self._rebuild_d1_order()

    def _split_d1_block(self, block_idx):
        """Split a D1 block to keep size <= M (naive half split)."""
        blk = self.D1[block_idx]
        if len(blk) <= self.M:
            return
        elems = blk.elements
        mid = len(elems) // 2
        left_e  = elems[:mid+1]
        right_e = elems[mid+1:]

        left  = Block(is_d1=True, upper_bound=(left_e[-1][1] if left_e else _as_tuple(-INF)))
        left.elements = left_e
        right = Block(is_d1=True, upper_bound=(right_e[-1][1] if right_e else _as_tuple(-INF)))
        right.elements = right_e

        self.D1.pop(block_idx)
        if left.elements:
            self.D1.insert(block_idx, left)
            block_idx += 1
        if right.elements:
            self.D1.insert(block_idx, right)

        self._rebuild_d1_order()
        for b in (left, right):
            for (k, v) in b.elements:
                self.key_map[k] = (v, b)

    # ---------------- public API ----------------

    def insert(self, key, value):
        """Insert/decrease-key to D1. `value` may be a scalar or a tuple."""
        vt = _as_tuple(value)
        cur = self.key_map.get(key)
        if cur and vt >= cur[0]:
            return
        if cur:
            self._delete_entry(key)

        # ensure D1 exists
        self._ensure_d1_nonempty()

        # find D1 block with upper_bound >= vt (tuple bisect)
        idx = bisect.bisect_left(self.d1_upper_bounds, vt)
        if idx >= len(self.D1):
            # vt is larger than all bounds → create a new block at the end
            new_blk = Block(is_d1=True, upper_bound=vt)
            self.D1.append(new_blk)
            self.d1_upper_bounds.append(new_blk.upper_bound)
            idx = len(self.D1) - 1

        target = self.D1[idx]

        # insert in value order (tuple)
        vals = [e[1] for e in target.elements]
        pos = bisect.bisect_left(vals, vt)
        target.elements.insert(pos, (key, vt))
        self.key_map[key] = (vt, target)
        self.nnz += 1

        # update bound/order, split if needed
        target.upper_bound = target.elements[-1][1]
        self._rebuild_d1_order()

        # after reorder, get fresh index of target and split if oversized
        try:
            t_idx = self.D1.index(target)
        except ValueError:
            t_idx = None
        if t_idx is not None and len(target) > self.M:
            self._split_d1_block(t_idx)

    def batch_prepend(self, items):
        """Prepend blocks to D0 (values better than existing ones kept)."""
        if not items:
            return
        filtered = []
        for k, v in items:
            vt = _as_tuple(v)
            cur = self.key_map.get(k)
            if cur and vt >= cur[0]:
                continue
            if cur:
                self._delete_entry(k)
            filtered.append((k, vt))
        if not filtered:
            return

        # sort by value tuple (asc) and split into small blocks (<= ceil(M/2))
        filtered.sort(key=lambda kv: kv[1])
        block_size = (self.M + 1) // 2 or 1
        new_blocks = []
        for i in range(0, len(filtered), block_size):
            blk = Block(is_d1=False)
            blk.elements = filtered[i:i+block_size]
            new_blocks.append(blk)

        # prepend to D0
        self.D0 = new_blocks + self.D0
        for blk in new_blocks:
            for k, vt in blk.elements:
                self.key_map[k] = (vt, blk)
                self.nnz += 1

    def pull(self):
        """
        Return (Bi_scalar, Si_set).
        Naive: gather all items, sort globally by value tuple, take top M,
               remove them, and set Bi as min remaining value's first component.
        """
        # gather all
        all_items = []
        for blk in self.D0:
            all_items.extend(blk.elements)
        for blk in self.D1:
            all_items.extend(blk.elements)

        if not all_items:
            return self.B, set()

        # pick global min M
        all_items.sort(key=lambda kv: kv[1])
        chosen = all_items[:self.M]
        S_keys = {k for (k, _) in chosen}

        # remove chosen keys from DS
        for k in S_keys:
            self._delete_entry(k)

        # compute Bi as min remaining (first component)
        rem_min_tuple = None
        for blk in self.D0:
            for _, vt in blk.elements:
                rem_min_tuple = vt if rem_min_tuple is None else min(rem_min_tuple, vt)
        for blk in self.D1:
            for _, vt in blk.elements:
                rem_min_tuple = vt if rem_min_tuple is None else min(rem_min_tuple, vt)

        Bi = rem_min_tuple[0] if rem_min_tuple is not None else self.B
        return Bi, S_keys

    def __bool__(self):
        return self.nnz > 0


# Backward-compat alias if algorithms.py imports DataStructureD
DataStructureD = Lemma33DataStructure


if __name__ == "__main__":
    # quick sanity check
    ds = Lemma33DataStructure(M=3, B=INF)
    ds.insert("a", (10, INF, INF, "a"))
    ds.insert("b", (20, INF, INF, "b"))
    ds.insert("c", (15, INF, INF, "c"))
    print("D1:", [[(k, v[0]) for (k, v) in b.elements] for b in ds.D1])

    ds.batch_prepend([("d",(5,INF,INF,"d")),("e",(3,INF,INF,"e")),("f",(7,INF,INF,"f")),("g",(1,INF,INF,"g"))])
    print("D0:", [[(k, v[0]) for (k, v) in b.elements] for b in ds.D0])

    Bi, S = ds.pull()
    print("pull ->", Bi, S)
