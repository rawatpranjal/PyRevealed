#!/usr/bin/env python3
"""Multi-user benchmark: Rust Rayon batch vs Python ProcessPoolExecutor.

Tests the real Uber-scale use case: analyzing thousands of independent
users in parallel.
"""

import os
import time

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from rust_garp import check_garp_batch_rust
from pyrevealed import BehaviorLog, check_garp


CPU = os.cpu_count() or 1


def generate_cohort(n_users, T_range, N=5, seed=42):
    rng = np.random.default_rng(seed)
    prices_list = []
    quantities_list = []
    for i in range(n_users):
        T = int(rng.integers(T_range[0], T_range[1] + 1))
        p = np.ascontiguousarray(rng.random((T, N)) + 0.1, dtype=np.float64)
        q = np.ascontiguousarray(rng.random((T, N)) + 0.1, dtype=np.float64)
        prices_list.append(p)
        quantities_list.append(q)
    return prices_list, quantities_list


def _py_check_one(args):
    p, q = args
    log = BehaviorLog(cost_vectors=p, action_vectors=q)
    return check_garp(log).is_consistent


def bench_python_sequential(prices_list, quantities_list):
    t0 = time.perf_counter()
    results = [_py_check_one((p, q)) for p, q in zip(prices_list, quantities_list)]
    return (time.perf_counter() - t0) * 1000, results


def bench_python_parallel(prices_list, quantities_list, workers=None):
    args = list(zip(prices_list, quantities_list))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_py_check_one, args))
    return (time.perf_counter() - t0) * 1000, results


def bench_rust_batch(prices_list, quantities_list):
    t0 = time.perf_counter()
    results = check_garp_batch_rust(prices_list, quantities_list, 1e-10)
    return (time.perf_counter() - t0) * 1000, results


def main():
    print("=" * 90)
    print(f" MULTI-USER BENCHMARK: Rust Rayon vs Python ProcessPoolExecutor")
    print(f" CPU cores: {CPU}")
    print("=" * 90)

    # Warmup
    wp, wq = generate_cohort(10, (10, 20))
    bench_python_sequential(wp, wq)
    bench_rust_batch(wp, wq)
    print("  Warmup done.\n")

    configs = [
        ("100 users, T=20-50 (casual)",       100,  (20, 50)),
        ("100 users, T=50-200 (regular)",      100,  (50, 200)),
        ("500 users, T=20-100 (city batch)",   500,  (20, 100)),
        ("1000 users, T=20-100 (large)",       1000, (20, 100)),
        ("1000 users, T=50-200 (heavy)",       1000, (50, 200)),
        ("2000 users, T=20-100 (xlarge)",      2000, (20, 100)),
    ]

    hdr = (f"{'Config':>42} | {'Py Seq':>9} | {'Py Pool':>9} | "
           f"{'Rust Rayon':>10} | {'Rust/PyPool':>11}")
    print(hdr)
    print("-" * len(hdr))

    for label, n_users, T_range in configs:
        pl, ql = generate_cohort(n_users, T_range)

        t_seq, r_seq = bench_python_sequential(pl, ql)
        t_pool, r_pool = bench_python_parallel(pl, ql, workers=CPU)
        t_rust, r_rust = bench_rust_batch(pl, ql)

        # Verify correctness
        assert r_seq == r_pool, "Pool mismatch!"
        assert r_seq == r_rust, f"Rust mismatch! {sum(r_seq)} vs {sum(r_rust)}"

        ratio = f"{t_pool / t_rust:.1f}x" if t_rust > 0.1 else "inf"
        print(f"{label:>42} | {t_seq:>7.0f}ms | {t_pool:>7.0f}ms | "
              f"{t_rust:>8.0f}ms | {ratio:>11}")

    print("-" * len(hdr))

    # Throughput projections
    print()
    pl, ql = generate_cohort(1000, (20, 100))
    bench_rust_batch(pl, ql)  # warmup
    t_rust_1k, _ = bench_rust_batch(pl, ql)
    rust_per_sec = 1000 / (t_rust_1k / 1000)

    bench_python_parallel(pl, ql, workers=CPU)  # warmup
    t_pool_1k, _ = bench_python_parallel(pl, ql, workers=CPU)
    py_per_sec = 1000 / (t_pool_1k / 1000)

    print(f"THROUGHPUT ({CPU} cores, T=20-100):")
    print(f"  Rust Rayon:          {rust_per_sec:,.0f} users/sec")
    print(f"  Python ProcessPool:  {py_per_sec:,.0f} users/sec")
    print()

    for lbl, count in [
        ("City (100K)", 100_000),
        ("National 1% (950K)", 950_000),
        ("Full base (95M)", 95_000_000),
    ]:
        t_r = count / rust_per_sec
        t_p = count / py_per_sec

        def fmt(s):
            if s < 60:
                return f"{s:.0f}s"
            if s < 3600:
                return f"{s / 60:.1f} min"
            return f"{s / 3600:.1f} hrs"

        speedup = t_p / t_r if t_r > 0 else float("inf")
        print(f"  {lbl:>20}: Rust {fmt(t_r):>10}  |  Python {fmt(t_p):>10}  |  Rust {speedup:.1f}x faster")

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
