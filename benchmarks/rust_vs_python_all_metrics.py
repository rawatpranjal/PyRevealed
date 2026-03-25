#!/usr/bin/env python3
"""Head-to-head: Rust Engine vs Python sequential, all metrics."""

import os
import time
import numpy as np

from pyrevealed import BehaviorLog, check_garp, compute_aei, compute_mpi
from pyrevealed.algorithms.harp import check_harp
from pyrevealed.engine import Engine

CPU = os.cpu_count() or 1


def gen(n, t_range=(20, 100), k=5, seed=42):
    rng = np.random.default_rng(seed)
    users = []
    for _ in range(n):
        t = int(rng.integers(t_range[0], t_range[1] + 1))
        p = np.ascontiguousarray(rng.random((t, k)) + 0.1, dtype=np.float64)
        q = np.ascontiguousarray(rng.random((t, k)) + 0.1, dtype=np.float64)
        users.append((p, q))
    return users


def python_all_metrics(users):
    """Python sequential: GARP + CCEI + MPI + HARP per user."""
    for p, q in users:
        log = BehaviorLog(cost_vectors=p, action_vectors=q)
        g = check_garp(log)
        if not g.is_consistent:
            compute_aei(log, method="discrete")
            compute_mpi(log)
        check_harp(log)


def fmt(s):
    if s < 1:
        return f"{s * 1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} hrs"


def main():
    print("=" * 80)
    print(f" RUST vs PYTHON: All Metrics Head-to-Head ({CPU} cores)")
    print("=" * 80)

    # Warmup
    warm = gen(20)
    python_all_metrics(warm)
    Engine(metrics=["garp", "ccei", "mpi", "harp"]).analyze_arrays(warm)

    engine = Engine(metrics=["garp", "ccei", "mpi", "harp"], chunk_size=50000)

    print(f"\n{'Users':>8} | {'Python seq':>12} | {'Rust Rayon':>12} | "
          f"{'Speedup':>8} | {'Py/s':>8} | {'Rust/s':>8}")
    print("-" * 78)

    for n in [100, 500, 1000, 5000, 10000]:
        users = gen(n)

        # Python sequential
        t0 = time.perf_counter()
        python_all_metrics(users)
        t_py = time.perf_counter() - t0

        # Rust engine
        t0 = time.perf_counter()
        engine.analyze_arrays(users)
        t_rust = time.perf_counter() - t0

        speedup = t_py / t_rust
        py_rate = n / t_py
        rust_rate = n / t_rust

        print(f"{n:>8,} | {fmt(t_py):>12} | {fmt(t_rust):>12} | "
              f"{speedup:>7.0f}x | {py_rate:>6.0f}/s | {rust_rate:>6.0f}/s")

    # Projections from 10K data point
    users_10k = gen(10000)

    t0 = time.perf_counter()
    python_all_metrics(users_10k[:1000])
    t_py_1k = time.perf_counter() - t0
    py_rate = 1000 / t_py_1k

    t0 = time.perf_counter()
    engine.analyze_arrays(users_10k)
    t_rust_10k = time.perf_counter() - t0
    rust_rate = 10000 / t_rust_10k

    print()
    print(f"Throughput: Python {py_rate:.0f}/s  |  Rust {rust_rate:.0f}/s  |  "
          f"Speedup: {rust_rate / py_rate:.0f}x")
    print()
    print("PROJECTIONS (all 4 metrics: GARP + CCEI + MPI + HARP)")
    print("-" * 78)
    for label, count in [
        ("100K users", 100_000),
        ("1M users", 1_000_000),
        ("10M users", 10_000_000),
        ("95M users", 95_000_000),
    ]:
        t_p = count / py_rate
        t_r = count / rust_rate
        print(f"  {label:>12}: Python {fmt(t_p):>8}  |  "
              f"Rust {fmt(t_r):>8}  |  {t_p / t_r:.0f}x")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
