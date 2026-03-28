#!/usr/bin/env python3
"""PrefGraph v2 - 1,000,000 User Benchmark.

Demonstrates the Rust-powered Engine analyzing 1M users with GARP + CCEI
using streaming chunks and thread-local scratchpads.
"""

import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prefgraph.engine import Engine


def generate_users(n_users, t_range=(20, 100), k=5, seed=42):
    """Generate random user data (prices, quantities) pairs."""
    rng = np.random.default_rng(seed)
    users = []
    for _ in range(n_users):
        t = int(rng.integers(t_range[0], t_range[1] + 1))
        p = np.ascontiguousarray(rng.random((t, k)) + 0.1, dtype=np.float64)
        q = np.ascontiguousarray(rng.random((t, k)) + 0.1, dtype=np.float64)
        users.append((p, q))
    return users


def fmt_time(s):
    if s < 1: return f"{s*1000:.0f}ms"
    if s < 60: return f"{s:.1f}s"
    if s < 3600: return f"{s/60:.1f} min"
    return f"{s/3600:.1f} hrs"


def main():
    CPU = os.cpu_count() or 1
    N_USERS = 1_000_000
    T_RANGE = (20, 100)
    K = 5
    CHUNK = 50_000

    print("=" * 70)
    print(f" PrefGraph v2 - {N_USERS:,} User Benchmark")
    print("=" * 70)

    engine = Engine(metrics=["garp", "ccei"], chunk_size=CHUNK)
    print(f" Engine:    {engine}")
    print(f" Users:     {N_USERS:,}")
    print(f" T range:   {T_RANGE[0]}-{T_RANGE[1]} orders per user")
    print(f" K:         {K} item categories")
    print(f" Metrics:   GARP + CCEI")
    print(f" Threads:   {CPU}")
    print(f" Chunks:    {CHUNK:,} users per chunk")
    print("=" * 70)

    # Phase 1: Generate data in streaming chunks
    print(f"\n Phase 1: Streaming analysis ({N_USERS//CHUNK} chunks of {CHUNK:,})...")
    tracemalloc.start()
    t_start = time.perf_counter()

    all_results = []
    rng = np.random.default_rng(42)
    processed = 0

    while processed < N_USERS:
        batch_size = min(CHUNK, N_USERS - processed)

        # Generate this chunk
        t0 = time.perf_counter()
        chunk = []
        for _ in range(batch_size):
            t = int(rng.integers(T_RANGE[0], T_RANGE[1] + 1))
            p = np.ascontiguousarray(rng.random((t, K)) + 0.1, dtype=np.float64)
            q = np.ascontiguousarray(rng.random((t, K)) + 0.1, dtype=np.float64)
            chunk.append((p, q))
        gen_time = time.perf_counter() - t0

        # Analyze this chunk
        t0 = time.perf_counter()
        results = engine.analyze_arrays(chunk)
        analyze_time = time.perf_counter() - t0

        all_results.extend(results)
        processed += batch_size

        _, peak_mem = tracemalloc.get_traced_memory()
        print(f"   Chunk {processed//CHUNK:3d}/{N_USERS//CHUNK}: "
              f"gen={gen_time:.1f}s  analyze={analyze_time:.1f}s  "
              f"mem={peak_mem/1e6:.0f}MB  "
              f"({processed:,}/{N_USERS:,})")

        # Free chunk memory
        del chunk, results

    total_time = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Phase 2: Results
    print(f"\n Phase 2: Results")
    print("-" * 70)

    n_consistent = sum(1 for r in all_results if r.is_garp)
    cceis = np.array([r.ccei for r in all_results])
    times_us = np.array([r.compute_time_us for r in all_results])

    throughput = N_USERS / total_time

    print(f" Total time:        {fmt_time(total_time)}")
    print(f" Throughput:        {throughput:,.0f} users/sec")
    print(f" Peak memory:       {peak_mem/1e6:.0f} MB")
    print()
    print(f" GARP-consistent:   {n_consistent:,} ({n_consistent/N_USERS*100:.1f}%)")
    print()
    print(f" CCEI Distribution:")
    print(f"   Mean:    {cceis.mean():.4f}")
    print(f"   Median:  {np.median(cceis):.4f}")
    print(f"   Std:     {cceis.std():.4f}")
    print(f"   P10:     {np.percentile(cceis, 10):.3f}")
    print(f"   P90:     {np.percentile(cceis, 90):.3f}")
    print()
    print(f" Per-user compute time:")
    print(f"   Mean:    {times_us.mean():.0f} us")
    print(f"   Median:  {np.median(times_us):.0f} us")
    print(f"   P99:     {np.percentile(times_us, 99):.0f} us")

    # Projections
    print()
    print("-" * 70)
    print(f" Scaling Projections (at {throughput:,.0f} users/sec on {CPU} cores):")
    for label, count in [("10M users", 10_000_000), ("95M users", 95_000_000)]:
        t = count / throughput
        print(f"   {label}: {fmt_time(t)}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
