#!/usr/bin/env python3
"""Champion vs Challenger: head-to-head performance comparison.

Runs current (champion) and optimized (challenger) algorithm variants on
identical data. Checks exact answer match and reports timing.

Three benchmarks:
  1. MPI: sparse predecessor lists (T=500, rationality=0.3, 200 users)
  2. HM:  no per-SCC adjacency copies (T=300, rationality=0.15, 100 users)
  3. Closure: u64 bitset DAG propagation (T=800, rationality=0.92, 50 users)
"""
import numpy as np
import sys

from prefgraph import generate_random_budgets
from prefgraph._rust_core import benchmark_mpi, benchmark_hm, benchmark_closure


def run_mpi_bench():
    """MPI: Karp's algorithm with dense vs sparse inner loop."""
    print("=" * 60)
    print("BENCHMARK 1: MPI (dense vs sparse predecessors)")
    print("  T=500, rationality=0.3, 200 users")
    print("=" * 60)

    data = generate_random_budgets(
        n_users=200, n_obs=500, n_goods=5,
        rationality=0.3, seed=42,
    )

    total_us_champ = 0
    total_us_chall = 0
    max_diff = 0.0
    n_nonzero = 0

    for i, (p, q) in enumerate(data):
        champ, chall, us_c, us_v = benchmark_mpi(p, q, 1e-10)
        diff = abs(champ - chall)
        if diff > max_diff:
            max_diff = diff
        if diff > 1e-12:
            print(f"  MISMATCH user {i}: champion={champ:.15f}, "
                  f"challenger={chall:.15f}, diff={diff:.2e}")
            return False
        total_us_champ += us_c
        total_us_chall += us_v
        if champ > 0:
            n_nonzero += 1

    t_c = total_us_champ / 1e6
    t_v = total_us_chall / 1e6
    speedup = t_c / t_v if t_v > 0 else float("inf")

    print(f"  Users with MPI > 0: {n_nonzero}/{len(data)}")
    print(f"  Max diff:       {max_diff:.2e}")
    print(f"  Champion total: {t_c:.3f}s")
    print(f"  Challenger total: {t_v:.3f}s")
    print(f"  Speedup:        {speedup:.2f}x")
    print(f"  RESULT: {'PASS' if max_diff < 1e-12 else 'FAIL'}")
    print()
    return True


def run_hm_bench():
    """HM greedy: with vs without per-SCC adjacency copies."""
    print("=" * 60)
    print("BENCHMARK 2: HM Greedy (per-SCC copies vs direct scoring)")
    print("  T=300, rationality=0.15, 100 users")
    print("=" * 60)

    data = generate_random_budgets(
        n_users=100, n_obs=300, n_goods=5,
        rationality=0.15, seed=42,
    )

    total_us_champ = 0
    total_us_chall = 0
    mismatches = 0

    for i, (p, q) in enumerate(data):
        c_champ, c_chall, t_total, us_c, us_v = benchmark_hm(p, q, 1e-10)
        if c_champ != c_chall:
            print(f"  MISMATCH user {i}: champion={c_champ}/{t_total}, "
                  f"challenger={c_chall}/{t_total}")
            mismatches += 1
        total_us_champ += us_c
        total_us_chall += us_v

    t_c = total_us_champ / 1e6
    t_v = total_us_chall / 1e6
    speedup = t_c / t_v if t_v > 0 else float("inf")

    print(f"  Mismatches:     {mismatches}/{len(data)}")
    print(f"  Champion total: {t_c:.3f}s")
    print(f"  Challenger total: {t_v:.3f}s")
    print(f"  Speedup:        {speedup:.2f}x")
    print(f"  RESULT: {'PASS' if mismatches == 0 else 'FAIL'}")
    print()
    return mismatches == 0


def run_closure_bench():
    """Closure: Vec<bool> reachability vs u64 bitset propagation."""
    print("=" * 60)
    print("BENCHMARK 3: Closure (bool-per-element vs u64 bitsets)")
    print("  T=800, rationality=0.92, 50 users")
    print("=" * 60)

    data = generate_random_budgets(
        n_users=50, n_obs=800, n_goods=5,
        rationality=0.92, seed=42,
    )

    total_us_champ = 0
    total_us_chall = 0
    mismatches = 0

    for i, (p, q) in enumerate(data):
        match, us_c, us_v = benchmark_closure(p, q, 1e-10)
        if not match:
            print(f"  MISMATCH user {i}: closure matrices differ!")
            mismatches += 1
        total_us_champ += us_c
        total_us_chall += us_v

    t_c = total_us_champ / 1e6
    t_v = total_us_chall / 1e6
    speedup = t_c / t_v if t_v > 0 else float("inf")

    print(f"  Mismatches:     {mismatches}/{len(data)}")
    print(f"  Champion total: {t_c:.3f}s")
    print(f"  Challenger total: {t_v:.3f}s")
    print(f"  Speedup:        {speedup:.2f}x")
    print(f"  RESULT: {'PASS' if mismatches == 0 else 'FAIL'}")
    print()
    return mismatches == 0


if __name__ == "__main__":
    print("PrefGraph Performance Audit: Champion vs Challenger")
    print("Each benchmark verifies exact answer parity + measures speedup.\n")

    results = []
    results.append(("MPI sparse", run_mpi_bench()))
    results.append(("HM greedy", run_hm_bench()))
    results.append(("Closure bitset", run_closure_bench()))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} {status}")
        if not passed:
            all_pass = False

    sys.exit(0 if all_pass else 1)
