import random
import sys
# Assuming your file is named D.py
from D import DataStructureD

# --- Debugging Wrapper Class ---
class DebugDataStructureD(DataStructureD):
    """
    A corrected wrapper for DataStructureD. It only verifies state after
    a complete top-level operation and adds detailed debugging for pull.
    """
    def __init__(self, M, B, ground_truth_ref, op_log_ref):
        super().__init__(M, B)
        self.ground_truth = ground_truth_ref
        self.op_log = op_log_ref
        # To store the last calculated threshold for debugging
        self.last_actual_threshold = None

    def _verify_state(self, operation_name):
        """
        Compares the internal state (nnz and all keys) against the ground truth.
        Raises an error with a detailed report if there is a mismatch.
        """
        if self.nnz != len(self.ground_truth):
            print("\n--- !! STATE INCONSISTENCY DETECTED (nnz) !! ---")
            print(f"Error occurred after operation: {operation_name}")
            print(f"Internal nnz ({self.nnz}) does not match ground truth size ({len(self.ground_truth)})")
            print("\nOperation Log that led to failure:")
            for op in self.op_log[-15:]:
                print(f"  - {op}")
            raise RuntimeError("State verification failed: nnz mismatch.")

        internal_keys = set(self.node_map.keys())
        truth_keys = set(self.ground_truth.keys())

        if internal_keys != truth_keys:
            print("\n--- !! STATE INCONSISTENCY DETECTED (keys) !! ---")
            print(f"Error occurred after operation: {operation_name}")
            missing = truth_keys - internal_keys
            extra = internal_keys - truth_keys
            if missing: print(f"Keys MISSING from data structure: {missing}")
            if extra:   print(f"Keys EXTRA in data structure: {extra}")
            print("\nOperation Log that led to failure:")
            for op in self.op_log[-15:]:
                print(f"  - {op}")
            raise RuntimeError("State verification failed: key mismatch.")

    def insert(self, key, value_tuple):
        super().insert(key, value_tuple)
        self._verify_state(f"insert(key={key}, value={value_tuple})")

    def batch_prepend(self, items):
        super().batch_prepend(items)
        self._verify_state(f"batch_prepend with {len(items)} items")

    def pull(self):
        # --- MODIFIED: Capture the actual threshold before the real pull ---
        collected, count = [], 0
        current_block = self.d0_head
        while current_block:
            collected.append(current_block)
            count += current_block.size
            if count >= self.M: break
            current_block = current_block.next_block
        if count < self.M:
            for _, blk in self.D1.items():
                collected.append(blk)
                count += blk.size
                if count >= self.M: break
        
        items = []
        for blk in collected:
            n = blk.head
            while n:
                items.append((n.key, n.value))
                n = n.next
        
        if len(items) > self.M:
            self.last_actual_threshold = self.quickselect(items, self.M - 1)
        else:
            self.last_actual_threshold = None

        # Now, call the original pull method
        result = super().pull()
        return result

# --- Main Debugging Function ---
def main():
    print("--- Running Randomized Debugger (Mixed Operations) ---")
    random.seed(100)  # Use the same seed that caused the failure

    num_operations = 10000
    M = 100
    B_tuple = (float('inf'), float('inf'), float('inf'), float('inf'))
    
    ground_truth = {}
    operation_log = []
    d = DebugDataStructureD(M=M, B=B_tuple, ground_truth_ref=ground_truth, op_log_ref=operation_log)

    try:
        for i in range(num_operations):
            op_choices = ['insert'] * 5 + ['pull'] * 2 + ['batch_prepend']
            op = random.choice(op_choices)

            if op == 'insert':
                key = random.randint(1, 1000)
                dist = random.randint(1, 10000)
                depth = random.randint(1, 10)
                pred = random.randint(1, 1000)
                value_tuple = (dist, depth, pred, key)
                operation_log.append(f"insert(key={key}, value={value_tuple})")
                if key not in ground_truth or value_tuple < ground_truth[key]:
                    ground_truth[key] = value_tuple
                d.insert(key, value_tuple)

            elif op == 'pull' and bool(d):
                operation_log.append("pull()")
                
                # Capture pre-pull state for detailed error reporting
                pre_pull_d0 = []
                current_block = d.d0_head
                while current_block:
                    n = current_block.head
                    while n:
                        pre_pull_d0.append((n.key, n.value))
                        n = n.next
                    current_block = current_block.next_block
                
                pre_pull_d1 = []
                for _, blk in d.D1.items():
                    n = blk.head
                    while n:
                        pre_pull_d1.append((n.key, n.value))
                        n = n.next
                
                pre_pull_d0.sort(key=lambda item: item[1])
                pre_pull_d1.sort(key=lambda item: item[1])

                _, pulled_keys = d.pull()
                
                pull_size = min(M, len(ground_truth))
                sorted_truth = sorted(ground_truth.items(), key=lambda item: item[1])
                expected_keys = {item[0] for item in sorted_truth[:pull_size]}
                
                if pulled_keys != expected_keys:
                    # ... error reporting as before ...
                    sys.exit(1)

                for key in pulled_keys:
                    ground_truth.pop(key, None)
                d._verify_state("pull()")

            elif op == 'batch_prepend':
                batch_size = random.randint(1, 5)
                batch = []
                # Enforce that all values to prepend are strictly smaller than any existing value
                current_min = min(ground_truth.values()) if ground_truth else d.B
                while len(batch) < batch_size:
                    key = random.randint(1, 1000)
                    dist = random.randint(1, 50)
                    depth = random.randint(1, 10)
                    pred = random.randint(1, 1000)
                    value_tuple = (dist, depth, pred, key)
                    if value_tuple < current_min:
                        batch.append((key, value_tuple))

                operation_log.append(f"batch_prepend({len(batch)} items)")
                for k, v_tuple in batch:
                    if k not in ground_truth or v_tuple < ground_truth[k]:
                        ground_truth[k] = v_tuple
                d.batch_prepend(batch)

    except RuntimeError as e:
        print(f"\nDEBUGGER HALTED: {e}")
        sys.exit(1)

    print("\n✓ All mixed operations completed successfully without errors.")

if __name__ == '__main__':
    main()
