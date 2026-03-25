"""Quick start: Score 100 simulated users with the Rust engine."""

import numpy as np
from pyrevealed.engine import Engine

# Generate 100 users: random prices + quantities (20 observations, 5 goods each)
rng = np.random.RandomState(42)
users = [
    (rng.rand(20, 5).astype(np.float64) + 0.1,
     rng.rand(20, 5).astype(np.float64) + 0.1)
    for _ in range(100)
]

# Score every user on 5 metrics in one call
engine = Engine(metrics=["garp", "ccei", "mpi", "harp", "hm"])
results = engine.analyze_arrays(users)

# Print results as a table
print(f"{'User':>6} {'Consistent':>11} {'CCEI':>6} {'MPI':>6} {'HARP':>6} {'HM':>7}")
print("-" * 50)
for i, r in enumerate(results[:15]):
    hm = f"{r.hm_consistent}/{r.hm_total}" if r.hm_total > 0 else "n/a"
    print(f"{i:6d} {str(r.is_garp):>11} {r.ccei:6.3f} {r.mpi:6.3f} {str(r.is_harp):>6} {hm:>7}")
print(f"... ({len(results)} total users)\n")

# Summary statistics
n_consistent = sum(1 for r in results if r.is_garp)
avg_ccei = np.mean([r.ccei for r in results])
avg_mpi = np.mean([r.mpi for r in results])
print(f"Consistent:  {n_consistent}/{len(results)} ({100*n_consistent/len(results):.0f}%)")
print(f"Avg CCEI:    {avg_ccei:.3f}")
print(f"Avg MPI:     {avg_mpi:.3f}")
print(f"Backend:     {engine.backend}")
