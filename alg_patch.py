# alg_patch.py
"""
Profile/ops-count patch for algorithms.py, applied non-invasively.

How it works:
- Subclasses algorithms.DataStructureD to count/time insert/pull/batch_prepend.
- Monkey-patches algorithms.{transformGraph, needsUpdate, update, findPivots, BaseCase, BMSSP}
  with wrappers that time/measure and then call the originals.
- Temporarily wraps heapq.heappush/pop inside BaseCase to count heap traffic.
- Exposes algorithms.get_prof() / algorithms.reset_prof() so your benchmark can pull stats.

Usage:
    import algorithms as alg
    import alg_patch  # <-- patches 'alg' in-place

Your code can now call:
    alg.reset_prof(); ... run stuff ...; prof = alg.get_prof()
"""

from __future__ import annotations
import time
import types
import heapq
import sys

# --- import the target module (already imported or import now) ---
import algorithms as alg

# ----------------------------- profiler store -----------------------------
PROF = {}
_TIC = {}

def _prof_reset():
    PROF.clear()
    # times
    for k in [
        "time.transform", "time.findPivots", "time.BaseCase", "time.BMSSP",
        "time.D.insert", "time.D.pull", "time.D.batch_prepend",
    ]:
        PROF[k] = 0.0
    # counts
    for k in [
        "count.findPivots.calls", "count.BaseCase.calls", "count.BMSSP.calls",
        "count.edges_scanned",            # approximated via needsUpdate calls that pass bound checks
        "count.relax_attempt", "count.needsUpdate", "count.update", "count.relax_success",
        "count.heap_push", "count.heap_pop",
        "count.W_total", "count.P_selected",
        "count.D.insert", "count.D.pull", "count.D.batch_prepend", "count.batch_prepend.items",
        "count.BMSSP.iterations",         # inferred via D.pull calls
    ]:
        PROF[k] = 0

def _bump(k, inc=1):
    PROF[k] = PROF.get(k, 0) + inc

def _tic(k):
    _TIC[k] = time.perf_counter()

def _toc(k):
    t0 = _TIC.get(k)
    if t0 is not None:
        PROF[k] = PROF.get(k, 0.0) + (time.perf_counter() - t0)

def get_prof():
    return dict(PROF)

def reset_prof():
    _prof_reset()

_prof_reset()

# ---------------------- save originals we will wrap -----------------------
_orig_transformGraph = alg.transformGraph
_orig_needsUpdate    = alg.needsUpdate
_orig_update         = alg.update
_orig_findPivots     = alg.findPivots
_orig_BaseCase       = alg.BaseCase
_orig_BMSSP          = alg.BMSSP

# Save original DS class so we can subclass and super()
_OrigDataStructureD  = alg.DataStructureD

# ---------------------- counting/timing DS subclass -----------------------
class CountingDataStructureD(_OrigDataStructureD):
    def insert(self, *args, **kwargs):
        _bump("count.D.insert")
        _tic("time.D.insert")
        try:
            return super().insert(*args, **kwargs)
        finally:
            _toc("time.D.insert")

    def pull(self, *args, **kwargs):
        _bump("count.D.pull")
        _bump("count.BMSSP.iterations")  # 1 pull ≈ 1 BMSSP while-iteration
        _tic("time.D.pull")
        try:
            return super().pull(*args, **kwargs)
        finally:
            _toc("time.D.pull")

    def batch_prepend(self, K, *args, **kwargs):
        _bump("count.D.batch_prepend")
        try:
            _bump("count.batch_prepend.items", len(K))
        except Exception:
            pass
        _tic("time.D.batch_prepend")
        try:
            return super().batch_prepend(K, *args, **kwargs)
        finally:
            _toc("time.D.batch_prepend")

# patch the class reference used inside algorithms.BMSSP
alg.DataStructureD = CountingDataStructureD

# ---------------------- function wrappers (monkey patch) -------------------
def _wrap_transformGraph():
    _tic("time.transform")
    try:
        return _orig_transformGraph()
    finally:
        _toc("time.transform")

def _wrap_needsUpdate(u, v):
    # Count attempts and calls
    _bump("count.relax_attempt")
    _bump("count.needsUpdate")
    return _orig_needsUpdate(u, v)

def _wrap_update(u, v):
    # Observe state to detect actual success
    _bump("count.update")
    before_d = alg.dist[v]
    before_depth = alg.depth[v]
    before_pred = alg.pred[v]
    res = _orig_update(u, v)
    # Detect success via changed triple
    if (alg.dist[v], alg.depth[v], alg.pred[v]) != (before_d, before_depth, before_pred):
        _bump("count.relax_success")
    return res

def _wrap_findPivots(B, S):
    _bump("count.findPivots.calls")
    _tic("time.findPivots")
    try:
        # We can't see inner edge scans without editing algorithms.py.
        # However, every relaxation attempt calls needsUpdate -> counted above.
        P, W = _orig_findPivots(B, S)
        # record sizes
        try:
            _bump("count.W_total", len(W))
        except Exception:
            pass
        try:
            _bump("count.P_selected", len(P))
        except Exception:
            pass
        return P, W
    finally:
        _toc("time.findPivots")

def _wrap_BaseCase(B, S):
    _bump("count.BaseCase.calls")
    _tic("time.BaseCase")
    # temporarily wrap heap ops
    orig_push, orig_pop = heapq.heappush, heapq.heappop

    def _hp_push(h, x):
        _bump("count.heap_push")
        return orig_push(h, x)

    def _hp_pop(h):
        _bump("count.heap_pop")
        return orig_pop(h)

    heapq.heappush = _hp_push
    heapq.heappop  = _hp_pop
    try:
        return _orig_BaseCase(B, S)
    finally:
        heapq.heappush = orig_push
        heapq.heappop  = orig_pop
        _toc("time.BaseCase")

def _wrap_BMSSP(l, B, S):
    _bump("count.BMSSP.calls")
    _tic("time.BMSSP")
    try:
        return _orig_BMSSP(l, B, S)
    finally:
        _toc("time.BMSSP")

# ---------------------- apply monkey patches to algorithms ------------------
alg.transformGraph = _wrap_transformGraph
alg.needsUpdate    = _wrap_needsUpdate
alg.update         = _wrap_update
alg.findPivots     = _wrap_findPivots
alg.BaseCase       = _wrap_BaseCase
alg.BMSSP          = _wrap_BMSSP

# expose profiler API on algorithms module so caller can do alg.get_prof()
alg.get_prof = get_prof
alg.reset_prof = reset_prof

# (Optional) register patched module for clarity
sys.modules.setdefault("alg_patch", sys.modules[__name__])
