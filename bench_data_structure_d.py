#!/usr/bin/env python3
import argparse
import math
import random
import time
from typing import List, Tuple

# Use your real implementation
import algorithms as alg  # for the tuple key shape (dist, depth, v)
from algorithms import DataStructureD


def now():
    return time.perf_counter()


def gen_key_under(B, v, dist_lo=0, dist_hi=10**9, depth_hi=10**6):
    """
    Generate a key (dist, depth, v) strictly less than B in lexicographic order.
    If B is (inf, inf, inf), any key works.
    """
    Bd, Bdep, Bv = B
    # If B is infinite-ish, just sample freely
    if math.isinf(Bd):
        d = random.randint(dist_lo, dist_hi)
        dep = random.randint(0, depth_hi)
        return (d, dep, v)

    # Otherwise ensure (d,dep,v) < (Bd, Bdep, Bv)
    # Pick d <= Bd, if d == Bd, pick dep < Bdep (if possible),
    # and if dep == Bdep, pick v < Bv.
    # Fall back to slightly smaller d if needed.
    d = random.randint(dist_lo, max(dist_lo, Bd))
    if d < Bd:
        dep = random.randint(0, depth_hi)
        return (d, dep, v)
    # d == Bd
    if Bdep > 0:
        dep = random.randint(0, Bdep - 1)
        return (d, dep, v)
    # dep must equal Bdep == 0; ensure v < Bv
    vv = v if v < Bv else max(0, Bv - 1)
    return (d, 0, vv)


def gen_key_between(lowB, highB, v, dist_lo=0, dist_hi=10**9, depth_hi=10**6):
    """
    Generate a key X such that lowB <= X < highB (lexicographic).
    Used to populate batch_prepend() between bounds returned by pulls.
    If constraints are tight, we relax heuristically.
    """
    ld, ldep, lv = lowB
    hd, hdep, hv = highB

    # If low is -inf-ish or equal to zero tuple, just target below highB
    if math.isinf(ld) and ld > 0:
        # (unlikely path)
        return gen_key_under(highB, v, dist_lo, dist_hi, depth_hi)

    # Try to pick a distance in [ld, hd], then enforce lex range
    if math.isinf(hd):
        # No upper cap → choose >= lowB freely
        d = max(ld, random.randint(dist_lo, dist_hi))
        dep = random.randint(0, depth_hi) if d > ld else max(ldep, 0)
        vv = v if (d, dep) > (ld, ldep) else max(v, lv)
        return (d, dep, vv)

    # hd finite: aim for hd-1 if we need strictness
    d_hi = max(ld, hd)
    if ld < hd:
        d = random.randint(ld, d_hi - 1)
        dep = random.randint(0, depth_hi) if d > ld else random.randint(ldep, max(ldep, depth_hi))
        vv = v
        return (d, dep, vv)
    else:
        # ld == hd: need dep/v room
        if ldep < hdep:
            dep = random.randint(ldep, hdep - 1)
            return (ld, dep, v)
        else:
            # ldep == hdep: use v band
            vv = v if v < hv else max(0, hv - 1)
            return (ld, ldep, vv)


def bench_data_structure(M: int,
                         B: Tuple[float, int, int],
                         V: int,
                         num_inserts: int,
                         num_pulls: int,
                         batch_size: int,
                         seed: int = 12345):
    """
    Microbenchmark for DataStructureD:
      - Construct D = DataStructureD(M, B)
      - Do 'num_inserts' inserts with keys < B
      - Then loop:
          pull() x 'num_pulls'
          after each pull, do batch_prepend of 'batch_size' items with keys in [B'_i, B_i)
    Returns timing and counts.
    """
    random.seed(seed)

    D = DataStructureD(M, B)

    # ------------------- phase 1: inserts -------------------
    t0 = now()
    for i in range(num_inserts):
        v = random.randrange(V)
        key = gen_key_under(B, v)
        D.insert(v, key)
    t1 = now()
    insert_time = t1 - t0

    # ------------------- phase 2: pull + batch_prepend -------
    pulls = 0
    batches = 0
    batch_items = 0

    pull_time = 0.0
    prepend_time = 0.0

    # We need a notion of the last B' to create in-interval keys.
    # Initialize last_Bp at something small to widen the range.
    last_Bp = (0, 0, 0)

    for _ in range(num_pulls):
        t2 = now()
        try:
            B_i, S_i = D.pull()
            pulls += 1
        except Exception:
            # If D is empty or pull not available, break
            break
        finally:
            pull_time += (now() - t2)

        # Generate batch_size items in [last_Bp, B_i)
        K = []
        for _ in range(batch_size):
            v = random.randrange(V)
            key = gen_key_between(last_Bp, B_i, v)
            K.append((v, key))

        t3 = now()
        D.batch_prepend(K)
        prepend_time += (now() - t3)
        batches += 1
        batch_items += len(K)
        last_Bp = B_i  # next round will use this as the low bound

    return {
        "M": M,
        "B": B,
        "V": V,
        "num_inserts": num_inserts,
        "num_pulls": pulls,
        "num_batches": batches,
        "batch_items": batch_items,
        "t_insert": insert_time,
        "t_pull": pull_time,
        "t_batch_prepend": prepend_time,
        "t_total": insert_time + pull_time + prepend_time,
    }


def pretty_print(res):
    def rate(n, t):
        return (n / t) if t > 0 else float('inf')

    ins_ns = (res["t_insert"] / max(1, res["num_inserts"])) * 1e6
    pull_ns = (res["t_pull"] / max(1, res["num_pulls"])) * 1e6
    prep_ns = (res["t_batch_prepend"] / max(1, res["num_batches"])) * 1e6
    per_item_prepend_us = (res["t_batch_prepend"] / max(1, res["batch_items"])) * 1e6

    print("\n=== DataStructureD Microbench ===")
    print(f"M={res['M']}, B={res['B']}, V={res['V']}")
    print(f"Inserts: {res['num_inserts']}  | Pulls: {res['num_pulls']}  | Batches: {res['num_batches']}  | Batch items: {res['batch_items']}")
    print(f"Time (s): insert={res['t_insert']:.6f}, pull={res['t_pull']:.6f}, batch_prepend={res['t_batch_prepend']:.6f}, total={res['t_total']:.6f}")
    print(f"Throughput (ops/sec): insert={rate(res['num_inserts'], res['t_insert']):.1f}, "
          f"pull={rate(res['num_pulls'], res['t_pull']):.1f}, "
          f"batch={rate(res['num_batches'], res['t_batch_prepend']):.1f}")
    print(f"Latency (μs/op): insert={ins_ns:.2f}, pull={pull_ns:.2f}, batch={prep_ns:.2f}, batch_prepend per-item={per_item_prepend_us:.2f}")


def main():
    ap = argparse.ArgumentParser(description="Microbenchmark for algorithms.DataStructureD")
    ap.add_argument("--M", type=int, default=1 << 16, help="D capacity parameter (matches BMSSP's 2^((l-1)*t))")
    ap.add_argument("--V", type=int, default=1 << 20, help="Universe of vertex ids for synthetic keys")
    ap.add_argument("--Bdist", type=float, default=float("inf"), help="Upper bound B.dist (float('inf') allowed)")
    ap.add_argument("--Bdepth", type=int, default=10**12, help="Upper bound B.depth")
    ap.add_argument("--Bv", type=int, default=10**12, help="Upper bound B.v")
    ap.add_argument("--inserts", type=int, default=500_000, help="# of insert operations")
    ap.add_argument("--pulls", type=int, default=50_000, help="# of pull operations to attempt")
    ap.add_argument("--batchSize", type=int, default=8, help="# items per batch_prepend")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    B = (args.Bdist, args.Bdepth, args.Bv)
    res = bench_data_structure(M=args.M,
                               B=B,
                               V=args.V,
                               num_inserts=args.inserts,
                               num_pulls=args.pulls,
                               batch_size=args.batchSize,
                               seed=args.seed)
    pretty_print(res)


if __name__ == "__main__":
    main()
