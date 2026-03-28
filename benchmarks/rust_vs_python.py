#!/usr/bin/env python3
"""Rust vs Python GARP benchmark.

Compares three implementations at various T values:
1. Python + SCC optimization (current production code)
2. Rust raw Floyd-Warshall (no SCC - pure compute comparison)
3. Rust + SCC (Tarjan's in Rust - full algorithmic + compute comparison)

Usage:
    # First build the Rust extension:
    cd benchmarks/rust_garp && pip install maturin && maturin develop --release && cd ../..

    # Then run:
    python3 benchmarks/rust_vs_python.py
"""

import sys
import time
import numpy as np

# Check if Rust extension is available
try:
    from rust_garp import check_garp_rust, check_garp_rust_scc
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from prefgraph import BehaviorLog, check_garp
from prefgraph.graph.transitive_closure import _floyd_warshall_direct


def bench_python_scc(prices, quantities, warmup=False):
    """Python GARP with SCC optimization (production code)."""
    log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)
    if warmup:
        check_garp(log)
        return 0, True
    t0 = time.perf_counter()
    result = check_garp(log)
    return (time.perf_counter() - t0) * 1000, result.is_consistent


def bench_python_raw_fw(prices, quantities, warmup=False):
    """Python raw Floyd-Warshall (no SCC - apples-to-apples vs Rust raw FW)."""
    T = prices.shape[0]
    E = prices @ quantities.T
    own_exp = np.diag(E)
    tol = 1e-10
    R = own_exp[:, np.newaxis] >= E - tol
    P = own_exp[:, np.newaxis] > E + tol
    np.fill_diagonal(P, False)
    if warmup:
        if T <= 500:
            _floyd_warshall_direct(R)
        return 0, True
    t0 = time.perf_counter()
    R_star = _floyd_warshall_direct(R)
    violation = np.any(R_star & P.T)
    elapsed = (time.perf_counter() - t0) * 1000
    return elapsed, not violation


def bench_rust_raw(prices, quantities, warmup=False):
    """Rust raw Floyd-Warshall (no SCC)."""
    if not HAS_RUST:
        return None, None
    if warmup:
        check_garp_rust(prices, quantities, 1e-10)
        return 0, True
    t0 = time.perf_counter()
    consistent = check_garp_rust(prices, quantities, 1e-10)
    return (time.perf_counter() - t0) * 1000, consistent


def bench_rust_scc(prices, quantities, warmup=False):
    """Rust with Tarjan's SCC + Floyd-Warshall."""
    if not HAS_RUST:
        return None, None
    if warmup:
        check_garp_rust_scc(prices, quantities, 1e-10)
        return 0, True
    t0 = time.perf_counter()
    consistent = check_garp_rust_scc(prices, quantities, 1e-10)
    return (time.perf_counter() - t0) * 1000, consistent


def main():
    print("=" * 80)
    print(" RUST vs PYTHON GARP BENCHMARK")
    print("=" * 80)

    if not HAS_RUST:
        print("\n  Rust extension not found! Build it first:")
        print("    cd benchmarks/rust_garp && pip install maturin && maturin develop --release")
        print("\n  Running Python-only benchmarks...\n")

    # Warmup JIT/caches
    print("  Warming up JIT compilers...")
    wp = np.random.rand(20, 5) + 0.1
    wq = np.random.rand(20, 5) + 0.1
    bench_python_scc(wp, wq, warmup=True)
    bench_python_raw_fw(wp, wq, warmup=True)
    if HAS_RUST:
        bench_rust_raw(wp, wq, warmup=True)
        bench_rust_scc(wp, wq, warmup=True)
    print("  Done.\n")

    # Benchmark sizes
    test_sizes = [50, 100, 200, 500, 1000, 2000]

    if HAS_RUST:
        header = (f"{'T':>6} | {'Py+SCC':>10} | {'Py RawFW':>10} | "
                  f"{'Rust RawFW':>10} | {'Rust+SCC':>10} | "
                  f"{'Rust/Py FW':>11} | {'Rust/Py SCC':>12} | {'Consistent':>10}")
    else:
        header = (f"{'T':>6} | {'Py+SCC':>10} | {'Py RawFW':>10} | "
                  f"{'SCC speedup':>12} | {'Consistent':>10}")

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for T in test_sizes:
        np.random.seed(42)
        N = 5
        prices = np.ascontiguousarray(np.random.rand(T, N) + 0.1, dtype=np.float64)
        quantities = np.ascontiguousarray(np.random.rand(T, N) + 0.1, dtype=np.float64)

        # Python + SCC
        t_py_scc, c_py_scc = bench_python_scc(prices, quantities)

        # Python raw FW (skip for T > 1000 - too slow)
        if T <= 1000:
            t_py_raw, c_py_raw = bench_python_raw_fw(prices, quantities)
        else:
            t_py_raw, c_py_raw = None, None

        if HAS_RUST:
            # Rust raw FW
            if T <= 2000:
                t_rust_raw, c_rust_raw = bench_rust_raw(prices, quantities)
            else:
                t_rust_raw, c_rust_raw = None, None

            # Rust + SCC
            t_rust_scc, c_rust_scc = bench_rust_scc(prices, quantities)

            # Verify correctness
            if c_rust_raw is not None and c_py_scc is not None:
                assert c_rust_raw == c_py_scc, f"Mismatch at T={T}!"
            if c_rust_scc is not None and c_py_scc is not None:
                assert c_rust_scc == c_py_scc, f"SCC mismatch at T={T}!"

            # Speedup ratios
            fw_ratio = f"{t_py_raw / t_rust_raw:.1f}x" if t_py_raw and t_rust_raw else "N/A"
            scc_ratio = f"{t_py_scc / t_rust_scc:.1f}x" if t_py_scc and t_rust_scc and t_rust_scc > 0 else "N/A"

            py_raw_str = f"{t_py_raw:.0f}ms" if t_py_raw is not None else "skip"
            rust_raw_str = f"{t_rust_raw:.0f}ms" if t_rust_raw is not None else "skip"
            rust_scc_str = f"{t_rust_scc:.0f}ms" if t_rust_scc is not None else "skip"

            print(f"{T:>6} | {t_py_scc:>8.0f}ms | {py_raw_str:>10} | "
                  f"{rust_raw_str:>10} | {rust_scc_str:>10} | "
                  f"{fw_ratio:>11} | {scc_ratio:>12} | {c_py_scc!s:>10}")
        else:
            py_raw_str = f"{t_py_raw:.0f}ms" if t_py_raw is not None else "skip"
            scc_speedup = f"{t_py_raw / t_py_scc:.1f}x" if t_py_raw and t_py_scc else "N/A"
            print(f"{T:>6} | {t_py_scc:>8.0f}ms | {py_raw_str:>10} | "
                  f"{scc_speedup:>12} | {c_py_scc!s:>10}")

    print("-" * len(header))

    if HAS_RUST:
        print("""
  COLUMNS:
    Py+SCC     = Python with SCC optimization (production code)
    Py RawFW   = Python raw Floyd-Warshall / Numba (no SCC)
    Rust RawFW = Rust raw Floyd-Warshall (no SCC)
    Rust+SCC   = Rust with Tarjan's SCC + Floyd-Warshall
    Rust/Py FW = Speedup of Rust raw FW over Python raw FW
    Rust/Py SCC = Speedup of Rust+SCC over Python+SCC
""")
    else:
        print("""
  COLUMNS:
    Py+SCC     = Python with SCC optimization (production code)
    Py RawFW   = Python raw Floyd-Warshall / Numba (no SCC)
    SCC speedup = How much SCC helps over raw FW
""")

    print("=" * 80)


if __name__ == "__main__":
    main()
