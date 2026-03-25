"""Scale scoring: 10K-100K users, streaming chunks, production throughput."""

import argparse
import time
import numpy as np
from pyrevealed.engine import Engine


def generate_heterogeneous_users(n_users, seed=42):
    """Simulate users with varying observation counts and goods."""
    rng = np.random.RandomState(seed)
    users = []
    for _ in range(n_users):
        n_obs = rng.randint(5, 50)   # 5-50 observations per user
        n_goods = rng.randint(3, 10)  # 3-10 goods
        p = rng.rand(n_obs, n_goods).astype(np.float64) + 0.1
        q = rng.rand(n_obs, n_goods).astype(np.float64) + 0.1
        users.append((p, q))
    return users


def main():
    parser = argparse.ArgumentParser(description="Score users at scale")
    parser.add_argument("--users", type=int, default=10_000, help="Number of users")
    parser.add_argument("--metrics", nargs="+", default=["garp", "ccei", "mpi", "harp", "hm"],
                        help="Metrics to compute")
    args = parser.parse_args()

    n = args.users
    print(f"Generating {n:,} heterogeneous users...")
    t0 = time.time()
    users = generate_heterogeneous_users(n)
    gen_ms = (time.time() - t0) * 1000
    print(f"  Generated in {gen_ms:.0f}ms")

    avg_obs = np.mean([p.shape[0] for p, _ in users])
    avg_goods = np.mean([p.shape[1] for p, _ in users])
    print(f"  Avg: {avg_obs:.0f} obs x {avg_goods:.0f} goods per user\n")

    engine = Engine(metrics=args.metrics, chunk_size=10_000)
    print(f"Engine: {engine}")
    print(f"Scoring {n:,} users on {len(args.metrics)} metrics...\n")

    t0 = time.time()
    results = engine.analyze_arrays(users)
    elapsed = time.time() - t0

    # Summary
    n_consistent = sum(1 for r in results if r.is_garp)
    avg_ccei = np.mean([r.ccei for r in results if r.ccei >= 0])
    avg_mpi = np.mean([r.mpi for r in results])
    throughput = n / elapsed

    print(f"{'Metric':<20} {'Value':>12}")
    print("-" * 35)
    print(f"{'Users scored':<20} {n:>12,}")
    print(f"{'Wall time':<20} {elapsed:>11.1f}s")
    print(f"{'Throughput':<20} {throughput:>10,.0f}/s")
    print(f"{'Consistent':<20} {n_consistent:>12,} ({100*n_consistent/n:.1f}%)")
    print(f"{'Avg CCEI':<20} {avg_ccei:>12.3f}")
    print(f"{'Avg MPI':<20} {avg_mpi:>12.3f}")
    print(f"{'Metrics':<20} {', '.join(args.metrics):>12}")
    print(f"{'Backend':<20} {engine.backend:>12}")


if __name__ == "__main__":
    main()
