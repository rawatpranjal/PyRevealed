#!/usr/bin/env python3
"""Decompose WHERE Python's multi-user time goes vs Rust."""

import os
import pickle
import time

import numpy as np

from pyrevealed import BehaviorLog, check_garp
from concurrent.futures import ProcessPoolExecutor


def _dummy(x):
    return x


def _garp_one(args):
    p, q = args
    return check_garp(BehaviorLog(cost_vectors=p, action_vectors=q)).is_consistent


def main():
    CPU = os.cpu_count() or 1
    np.random.seed(42)
    p = np.random.rand(50, 5) + 0.1
    q = np.random.rand(50, 5) + 0.1

    # Warmup JIT
    log = BehaviorLog(cost_vectors=p, action_vectors=q)
    check_garp(log)

    print("=" * 65)
    print(" WHERE DOES PYTHON'S TIME GO? (per-user T=50)")
    print("=" * 65)

    # 1. Pure GARP compute
    times = []
    for _ in range(200):
        log = BehaviorLog(cost_vectors=p, action_vectors=q)
        t0 = time.perf_counter()
        check_garp(log)
        times.append((time.perf_counter() - t0) * 1000)
    garp_ms = np.median(times)
    print(f"  Pure check_garp compute:    {garp_ms:.2f}ms")

    # 2. ProcessPool spawn+destroy overhead
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=CPU) as ex:
        list(ex.map(_dummy, range(100)))
    pool_ms = (time.perf_counter() - t0) * 1000
    print(f"  ProcessPool overhead:       {pool_ms:.0f}ms  (spawn {CPU} processes)")

    # 3. Pickle round-trip per user
    t0 = time.perf_counter()
    for _ in range(10000):
        data = pickle.dumps((p, q))
        pickle.loads(data)
    pickle_ms = (time.perf_counter() - t0) / 10000 * 1000
    print(f"  Pickle round-trip/user:     {pickle_ms:.3f}ms  ({len(data)} bytes)")

    # 4. BehaviorLog construction
    t0 = time.perf_counter()
    for _ in range(10000):
        BehaviorLog(cost_vectors=p, action_vectors=q)
    obj_ms = (time.perf_counter() - t0) / 10000 * 1000
    print(f"  BehaviorLog construction:   {obj_ms:.3f}ms")

    # 5. Per-process Python interpreter + import overhead
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=1) as ex:
        # First call forces import of pyrevealed in child
        ex.submit(_garp_one, (p, q)).result()
    first_call_ms = (time.perf_counter() - t0) * 1000
    print(f"  First call (import in child): {first_call_ms:.0f}ms")

    print()
    print("=" * 65)
    print(" THE MATH: 1000 users, T=50")
    print("=" * 65)

    n = 1000
    compute = n * garp_ms
    pickling = n * pickle_ms * 2  # send + receive
    pool = pool_ms
    total = compute + pickling + pool

    print(f"  GARP compute ({n} x {garp_ms:.1f}ms):  {compute:.0f}ms")
    print(f"  Pickling ({n} x {pickle_ms:.2f}ms x 2): {pickling:.0f}ms")
    print(f"  Pool spawn/destroy:            {pool:.0f}ms")
    print(f"  ---")
    print(f"  Total estimated:               {total:.0f}ms")
    print(f"  With {CPU} cores parallel:       ~{total/CPU:.0f}ms (compute) + {pickling + pool:.0f}ms (serial overhead)")
    print()
    print(f"  Rust does the same {n} users in ~50ms because:")
    print(f"    - No pickling    (shared memory, zero copy)")
    print(f"    - No pool spawn  (Rayon threadpool, always warm)")
    print(f"    - No interpreter (no Python in the hot loop)")
    print(f"    - No BehaviorLog (operates on raw arrays)")
    print()

    # 6. Actual measured comparison
    print("=" * 65)
    print(" ACTUAL MEASUREMENT: 1000 users, T=20-100")
    print("=" * 65)

    rng = np.random.default_rng(42)
    users = []
    for i in range(n):
        T = int(rng.integers(20, 101))
        up = np.ascontiguousarray(rng.random((T, 5)) + 0.1, dtype=np.float64)
        uq = np.ascontiguousarray(rng.random((T, 5)) + 0.1, dtype=np.float64)
        users.append((up, uq))

    # Sequential Python
    t0 = time.perf_counter()
    seq_results = [_garp_one(u) for u in users]
    t_seq = (time.perf_counter() - t0) * 1000

    # Parallel Python
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=CPU) as ex:
        pool_results = list(ex.map(_garp_one, users))
    t_pool = (time.perf_counter() - t0) * 1000

    # Rust
    from rust_garp import check_garp_batch_rust
    pl = [u[0] for u in users]
    ql = [u[1] for u in users]
    check_garp_batch_rust(pl, ql, 1e-10)  # warmup
    t0 = time.perf_counter()
    rust_results = check_garp_batch_rust(pl, ql, 1e-10)
    t_rust = (time.perf_counter() - t0) * 1000

    print(f"  Python sequential:    {t_seq:.0f}ms")
    print(f"  Python ProcessPool:   {t_pool:.0f}ms  (parallelism helps {t_seq/t_pool:.1f}x)")
    print(f"  Rust Rayon:           {t_rust:.0f}ms  ({t_pool/t_rust:.0f}x faster than Pool)")
    print()
    print(f"  Breakdown of Python Pool's {t_pool:.0f}ms:")
    pure_compute = t_seq / CPU
    overhead = t_pool - pure_compute
    print(f"    Compute ({CPU} cores):   ~{pure_compute:.0f}ms")
    print(f"    IPC overhead:        ~{overhead:.0f}ms  ({overhead/t_pool*100:.0f}% of total!)")
    print(f"    Rust avoids ALL of that overhead.")
    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
