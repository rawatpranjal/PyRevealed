#!/usr/bin/env python3
"""Memory-optimized Rust benchmark: scratchpads + bitpacking + streaming chunks.

Compares:
- Original Rust batch (allocates per user)
- Optimized Rust batch (thread-local scratchpads + bitpacked closure)
- Streaming chunks (process in 10K-user windows, bounded memory)
"""

import os
import sys
import time
import tracemalloc

import numpy as np

from rust_garp import check_garp_batch_rust, check_garp_batch_optimized

CPU = os.cpu_count() or 1


def generate_chunk(n_users, T_range, N=5, seed=0):
    rng = np.random.default_rng(seed)
    pl, ql = [], []
    for _ in range(n_users):
        T = int(rng.integers(T_range[0], T_range[1] + 1))
        pl.append(np.ascontiguousarray(rng.random((T, N)) + 0.1, dtype=np.float64))
        ql.append(np.ascontiguousarray(rng.random((T, N)) + 0.1, dtype=np.float64))
    return pl, ql


def bench_batch(func, pl, ql, tolerance=1e-10):
    t0 = time.perf_counter()
    results = func(pl, ql, tolerance)
    elapsed = time.perf_counter() - t0
    return elapsed, results


def bench_streaming(func, total_users, chunk_size, T_range, N=5, tolerance=1e-10):
    """Process users in streaming chunks - memory bounded to chunk_size users."""
    all_results = []
    t0 = time.perf_counter()
    processed = 0

    while processed < total_users:
        batch = min(chunk_size, total_users - processed)
        pl, ql = generate_chunk(batch, T_range, N, seed=processed)
        results = func(pl, ql, tolerance)
        all_results.extend(results)
        processed += batch
        # pl, ql are dropped here - memory freed

    elapsed = time.perf_counter() - t0
    return elapsed, all_results


def fmt_time(s):
    if s < 1:
        return f"{s * 1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} hrs"


def fmt_mem(bytes_val):
    if bytes_val < 1024:
        return f"{bytes_val}B"
    if bytes_val < 1024 ** 2:
        return f"{bytes_val / 1024:.0f}KB"
    if bytes_val < 1024 ** 3:
        return f"{bytes_val / 1024 ** 2:.0f}MB"
    return f"{bytes_val / 1024 ** 3:.1f}GB"


def main():
    print("=" * 85)
    print(f" MEMORY-OPTIMIZED RUST BENCHMARK ({CPU} cores)")
    print(f" Original vs Scratchpad+Bitpack vs Streaming")
    print("=" * 85)

    # Warmup
    wp, wq = generate_chunk(100, (20, 50))
    check_garp_batch_rust(wp, wq, 1e-10)
    check_garp_batch_optimized(wp, wq, 1e-10)
    print("  Warmup done.\n")

    # --- Part 1: Speed comparison (original vs optimized) ---
    print("PART 1: SPEED - Original vs Optimized (T=20-100)")
    print("-" * 85)
    hdr = f"{'Users':>10} | {'Original':>10} | {'Optimized':>10} | {'Speedup':>8} | {'Correct':>8}"
    print(hdr)
    print("-" * 85)

    for n in [1000, 5000, 10000, 50000, 100000]:
        pl, ql = generate_chunk(n, (20, 100))

        t_orig, r_orig = bench_batch(check_garp_batch_rust, pl, ql)
        t_opt, r_opt = bench_batch(check_garp_batch_optimized, pl, ql)

        correct = r_orig == r_opt
        speedup = t_orig / t_opt if t_opt > 0 else float("inf")
        print(f"{n:>10,} | {fmt_time(t_orig):>10} | {fmt_time(t_opt):>10} | {speedup:>7.1f}x | {'OK' if correct else 'FAIL':>8}")

    # --- Part 2: Memory comparison ---
    print()
    print("PART 2: MEMORY - All-at-once vs Streaming chunks")
    print("-" * 85)

    for n_users, chunk_size in [(10000, 10000), (50000, 10000), (100000, 10000)]:
        # All at once - measure peak memory
        tracemalloc.start()
        pl, ql = generate_chunk(n_users, (20, 100))
        check_garp_batch_optimized(pl, ql, 1e-10)
        _, peak_all = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Streaming - measure peak memory
        tracemalloc.start()
        bench_streaming(check_garp_batch_optimized, n_users, chunk_size, (20, 100))
        _, peak_stream = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        reduction = peak_all / peak_stream if peak_stream > 0 else float("inf")
        print(f"  {n_users:>10,} users: all-at-once {fmt_mem(peak_all):>8} | "
              f"streaming (chunks of {chunk_size:,}) {fmt_mem(peak_stream):>8} | "
              f"{reduction:.1f}x reduction")

    # --- Part 3: Stress test - streaming to extreme scale ---
    print()
    print("PART 3: EXTREME SCALE - Streaming (chunks of 10K)")
    print("-" * 85)
    print(f"{'Total Users':>14} | {'Time':>10} | {'Users/sec':>12} | {'Peak Mem':>10}")
    print("-" * 85)

    for total in [10000, 50000, 100000, 500000, 1000000]:
        tracemalloc.start()
        elapsed, results = bench_streaming(
            check_garp_batch_optimized, total, 10000, (20, 100)
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        per_sec = total / elapsed
        print(f"{total:>14,} | {fmt_time(elapsed):>10} | {per_sec:>10,.0f}/s | {fmt_mem(peak):>10}")

    # Projections
    print()
    print("-" * 85)
    # Use the 100K number for projections
    tracemalloc.start()
    elapsed_100k, _ = bench_streaming(
        check_garp_batch_optimized, 100000, 10000, (20, 100)
    )
    _, peak_100k = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rate = 100000 / elapsed_100k
    print(f"\nPROJECTIONS (at {rate:,.0f} users/sec, peak memory {fmt_mem(peak_100k)}):")
    for label, count in [
        ("City (100K)", 100_000),
        ("National 1% (950K)", 950_000),
        ("Full Uber (10M)", 10_000_000),
        ("Full Uber (95M)", 95_000_000),
    ]:
        t = count / rate
        print(f"  {label:>22}: {fmt_time(t):>10}  (memory stays at ~{fmt_mem(peak_100k)})")

    print()
    print("=" * 85)


if __name__ == "__main__":
    main()
