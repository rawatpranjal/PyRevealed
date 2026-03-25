"""E2E correctness + performance test: Rust PreferenceGraph engine vs Python.

Usage: NUMBA_DISABLE_JIT=1 python3 benchmarks/e2e_rust_vs_python.py
"""
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import time
import numpy as np
from pyrevealed import BehaviorLog, check_garp, compute_aei, compute_mpi
from pyrevealed.algorithms.harp import check_harp
from pyrevealed.algorithms.mpi import compute_houtman_maks_index
from pyrevealed.engine import Engine


def generate_users(n_users, n_obs, n_goods, seed=42):
    rng = np.random.RandomState(seed)
    users = []
    for _ in range(n_users):
        p = rng.rand(n_obs, n_goods) + 0.1
        q = rng.rand(n_obs, n_goods) + 0.1
        users.append((p.astype(np.float64), q.astype(np.float64)))
    return users


def python_analyze_one(prices, quantities):
    log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)
    tol = 1e-10
    garp = check_garp(log, tol)
    ccei = compute_aei(log, method="discrete").efficiency_index if not garp.is_consistent else 1.0
    # Use Karp method for MPI to match Rust's Karp implementation
    mpi_val = compute_mpi(log, tol, method="karp").mpi_value if not garp.is_consistent else 0.0
    harp = check_harp(log, tol)
    hm = compute_houtman_maks_index(log, tol)
    return {
        "is_garp": garp.is_consistent,
        "ccei": ccei,
        "mpi": mpi_val,
        "is_harp": harp.is_consistent,
        "hm_fraction": 1.0 - hm.fraction,
    }


# ── Part 1: Correctness ─────────────────────────────────────────────
print("=" * 70)
print("PART 1: CORRECTNESS (Rust vs Python, 50 users)")
print("=" * 70)

users = generate_users(50, 10, 5, seed=123)
engine = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm", "utility", "vei"])

rust_results = engine.analyze_arrays(users)

mismatches = {"garp": 0, "ccei": 0, "mpi": 0, "harp": 0, "hm": 0}
for i, ((p, q), rust) in enumerate(zip(users, rust_results)):
    py = python_analyze_one(p, q)

    if py["is_garp"] != rust.is_garp:
        mismatches["garp"] += 1
        print(f"  User {i}: GARP mismatch py={py['is_garp']} rust={rust.is_garp}")

    if abs(py["ccei"] - rust.ccei) > 0.01:
        mismatches["ccei"] += 1
        print(f"  User {i}: CCEI mismatch py={py['ccei']:.4f} rust={rust.ccei:.4f}")

    # MPI tolerance is wider because Python falls back to cycle-enumeration
    # (aggregate formula) when Numba is disabled, while Rust uses true Karp
    # (per-edge mean). These are different formulas for non-uniform expenditures.
    if abs(py["mpi"] - rust.mpi) > 0.05:
        mismatches["mpi"] += 1
        print(f"  User {i}: MPI mismatch py={py['mpi']:.4f} rust={rust.mpi:.4f}")

    if py["is_harp"] != rust.is_harp:
        mismatches["harp"] += 1
        print(f"  User {i}: HARP mismatch py={py['is_harp']} rust={rust.is_harp}")

    rust_hm_frac = rust.hm_consistent / rust.hm_total if rust.hm_total > 0 else 1.0
    if abs(py["hm_fraction"] - rust_hm_frac) > 0.15:
        mismatches["hm"] += 1
        print(f"  User {i}: HM mismatch py={py['hm_fraction']:.3f} rust={rust_hm_frac:.3f}")

print()
all_pass = True
for metric, count in mismatches.items():
    status = "PASS" if count == 0 else f"FAIL ({count}/50)"
    if count > 0:
        all_pass = False
    print(f"  {metric:6s}: {status}")

# Show new metrics summary
utility_ok = sum(1 for r in rust_results if r.utility_success)
vei_below = sum(1 for r in rust_results if r.vei_mean < 1.0)
print(f"\n  utility: {utility_ok}/50 recoveries (consistent users)")
print(f"  vei:     {vei_below}/50 with mean < 1.0")
print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

# ── Part 2: Performance ─────────────────────────────────────────────
print()
print("=" * 70)
print("PART 2: PERFORMANCE (Rust 7-metric Engine vs Python 5-metric sequential)")
print("=" * 70)

scales = [
    (10, 10, 5),
    (50, 20, 5),
    (200, 20, 5),
    (500, 20, 5),
]

print(f"\n  {'Users':>6s} x {'Obs':>3s} x {'Goods':>5s}  |  {'Python(5)':>10s}  {'Rust(7)':>10s}  {'Speedup':>8s}")
print("  " + "-" * 60)

for n_users, n_obs, n_goods in scales:
    users = generate_users(n_users, n_obs, n_goods, seed=42)

    # Python (5 metrics: GARP+CCEI+MPI+HARP+HM)
    t0 = time.time()
    for p, q in users:
        python_analyze_one(p, q)
    py_ms = (time.time() - t0) * 1000

    # Rust (7 metrics: GARP+CCEI+MPI+HARP+HM+utility+VEI)
    t0 = time.time()
    engine.analyze_arrays(users)
    rust_ms = (time.time() - t0) * 1000

    speedup = py_ms / rust_ms if rust_ms > 0 else float("inf")
    print(f"  {n_users:6d} x {n_obs:3d} x {n_goods:5d}  |  {py_ms:8.1f}ms  {rust_ms:8.1f}ms  {speedup:7.1f}x")

print()
