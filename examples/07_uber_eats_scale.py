#!/usr/bin/env python3
"""Example: Uber Eats at Scale — Revealed Preference for Food Delivery.

Demonstrates PyRevealed on simulated Uber Eats order data with realistic
user heterogeneity and multi-core parallel processing.

Key features:
- Realistic food-delivery data simulation (sparse menus, log-normal prices)
- Heterogeneous users: casual (20 orders) to power users (800+ orders)
- Full pipeline: GARP + AEI + Houtman-Maks per user
- Parallel cohort analysis via concurrent.futures.ProcessPoolExecutor
- Production-ready scaling projections

Usage:
    python examples/07_uber_eats_scale.py
    python examples/07_uber_eats_scale.py --users 500 --workers 8
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pyrevealed import BehaviorLog, check_garp, compute_aei
from pyrevealed.algorithms.mpi import compute_houtman_maks_index


# =============================================================================
# Part 1: Uber Eats Data Simulator
# =============================================================================

# Realistic menu categories and price tiers
MENU_CATEGORIES = {
    "appetizers": (6.0, 2.0, 8),      # (mean_price, std, count)
    "salads": (11.0, 3.0, 6),
    "burgers": (14.0, 3.5, 10),
    "sandwiches": (12.0, 2.5, 8),
    "pizza": (16.0, 4.0, 10),
    "pasta": (15.0, 3.0, 6),
    "sushi_rolls": (13.0, 4.0, 12),
    "bowls": (14.0, 3.0, 8),
    "tacos": (10.0, 2.5, 8),
    "desserts": (8.0, 2.0, 6),
    "drinks": (4.0, 1.5, 10),
    "sides": (5.0, 1.5, 8),
}


def _build_menu(n_items: int, rng: np.random.Generator) -> np.ndarray:
    """Build a realistic restaurant menu with category-based pricing."""
    prices = np.zeros(n_items)
    idx = 0
    cats = list(MENU_CATEGORIES.values())

    while idx < n_items:
        mean_p, std_p, _ = cats[idx % len(cats)]
        price = rng.lognormal(np.log(mean_p), std_p / mean_p)
        prices[idx] = np.clip(price, 3.0, 55.0)
        idx += 1

    return prices


def simulate_uber_eats_user(
    user_id: str,
    n_orders: int,
    n_menu_items: int = 100,
    seed: int = 0,
    noise_level: float = 0.3,
) -> BehaviorLog:
    """
    Simulate one Uber Eats user's order history.

    Each observation = one order session. The user sees a menu of n_menu_items
    with fixed prices and picks 1-2 items. Underlying preferences follow a
    noisy utility model, creating realistic ~80% AEI.

    Args:
        user_id: Identifier for this user
        n_orders: Number of orders (observations)
        n_menu_items: Size of the restaurant menu (goods)
        seed: Random seed for reproducibility
        noise_level: How noisy the choices are (0=rational, 1=random)

    Returns:
        BehaviorLog with sparse quantity vectors
    """
    rng = np.random.default_rng(seed)
    N = n_menu_items

    # Build a realistic menu with category-structured prices
    menu_prices = _build_menu(N, rng)

    # User's latent preference weights (Dirichlet — some items strongly preferred)
    # Sparse preferences: user only really likes ~20% of the menu
    raw_alpha = rng.exponential(0.3, size=N)
    # Make preferences spiky — a few favorites, many ignored
    top_k = max(3, N // 5)
    top_items = rng.choice(N, size=top_k, replace=False)
    raw_alpha[top_items] *= rng.uniform(3.0, 10.0, size=top_k)
    alpha = raw_alpha / raw_alpha.sum()

    # Budget varies per order: some orders are solo lunch, some are group dinner
    budget_mean = rng.uniform(15.0, 35.0)  # User's typical spend
    budget_std = budget_mean * 0.3

    prices_matrix = np.zeros((n_orders, N))
    quantities_matrix = np.zeros((n_orders, N))

    for t in range(n_orders):
        # Prices vary per order: promotions, surge pricing, time-of-day
        # This is critical — identical prices across observations can't
        # create GARP violations. Real food delivery has 10-30% price
        # variation from promos, surge, and dynamic pricing.
        price_shock = rng.lognormal(0.0, 0.15, size=N)  # ~15% variation
        # Random flash discounts on 10-20% of items
        n_promo = rng.integers(N // 10, N // 5 + 1)
        promo_items = rng.choice(N, size=n_promo, replace=False)
        price_shock[promo_items] *= rng.uniform(0.6, 0.85, size=n_promo)
        current_prices = menu_prices * price_shock
        current_prices = np.clip(current_prices, 2.0, 60.0)
        prices_matrix[t] = current_prices

        # This order's budget
        budget = max(8.0, rng.normal(budget_mean, budget_std))

        # Utility-maximizing choice with noise
        # Cobb-Douglas: pick items with highest alpha_i / p_i ratio
        value_per_dollar = alpha / current_prices
        noise = rng.exponential(1.0, size=N)
        noisy_value = value_per_dollar * (1.0 - noise_level + noise_level * noise)

        # Pick 1-2 items that fit the budget
        n_items_ordered = rng.choice([1, 2], p=[0.6, 0.4])
        sorted_items = np.argsort(-noisy_value)

        spent = 0.0
        for item_idx in sorted_items:
            if spent + current_prices[item_idx] > budget * 1.1:
                continue
            qty = 1.0
            quantities_matrix[t, item_idx] = qty
            spent += current_prices[item_idx] * qty
            n_items_ordered -= 1
            if n_items_ordered <= 0:
                break

        # If nothing was ordered (rare edge case), pick the cheapest item
        if spent == 0:
            cheapest = np.argmin(current_prices)
            quantities_matrix[t, cheapest] = 1.0

    return BehaviorLog(
        cost_vectors=prices_matrix,
        action_vectors=quantities_matrix,
        user_id=user_id,
    )


# User archetypes with realistic order count distributions
USER_ARCHETYPES = {
    "churned":     {"weight": 0.15, "orders": (5, 15),   "noise": (0.4, 0.7)},
    "casual":      {"weight": 0.30, "orders": (15, 60),  "noise": (0.3, 0.5)},
    "regular":     {"weight": 0.30, "orders": (60, 200), "noise": (0.2, 0.4)},
    "power_user":  {"weight": 0.15, "orders": (200, 500),"noise": (0.15, 0.35)},
    "super_power": {"weight": 0.10, "orders": (500, 1000),"noise": (0.1, 0.3)},
}


def simulate_uber_eats_cohort(
    n_users: int,
    n_menu_items: int = 100,
    seed: int = 42,
) -> list[BehaviorLog]:
    """
    Simulate a cohort of Uber Eats users with realistic heterogeneity.

    Users are drawn from archetypes: churned (few orders, noisy), casual,
    regular, power users, and super-power users. Order counts, noise levels,
    and menu sizes vary realistically.

    Args:
        n_users: Number of users to simulate
        n_menu_items: Base menu size (varies +/- 30% per user's restaurant mix)
        seed: Master random seed

    Returns:
        List of BehaviorLog objects, one per user
    """
    rng = np.random.default_rng(seed)
    logs: list[BehaviorLog] = []

    # Build archetype assignment probabilities
    archetypes = list(USER_ARCHETYPES.keys())
    weights = np.array([USER_ARCHETYPES[a]["weight"] for a in archetypes])
    weights /= weights.sum()

    for i in range(n_users):
        # Assign archetype
        arch_name = rng.choice(archetypes, p=weights)
        arch = USER_ARCHETYPES[arch_name]

        # Draw order count from archetype range
        lo, hi = arch["orders"]
        n_orders = int(rng.integers(lo, hi + 1))

        # Draw noise level from archetype range
        noise_lo, noise_hi = arch["noise"]
        noise = rng.uniform(noise_lo, noise_hi)

        # Vary menu size per user (different restaurant mixes)
        user_menu = int(n_menu_items * rng.uniform(0.7, 1.3))
        user_menu = max(20, min(user_menu, 250))

        log = simulate_uber_eats_user(
            user_id=f"user_{i:05d}",
            n_orders=n_orders,
            n_menu_items=user_menu,
            seed=seed + i * 1000,
            noise_level=noise,
        )
        logs.append(log)

    return logs


# =============================================================================
# Part 2: Single-User Analysis
# =============================================================================


@dataclass
class UserResult:
    """Results for one user's revealed preference analysis."""
    user_id: str
    n_orders: int
    n_menu_items: int
    is_consistent: bool
    aei: float
    hm_fraction: float
    garp_time_ms: float
    aei_time_ms: float
    hm_time_ms: float
    error: Optional[str] = None


def analyze_user(log: BehaviorLog) -> UserResult:
    """
    Run the full revealed preference pipeline on one user.

    Pipeline: GARP check -> AEI (if inconsistent) -> Houtman-Maks

    Args:
        log: User's order history as a BehaviorLog

    Returns:
        UserResult with all metrics and timings
    """
    user_id = log.user_id or "unknown"
    n_orders = log.num_records
    n_items = log.action_vectors.shape[1]

    try:
        # GARP consistency check
        t0 = time.perf_counter()
        garp = check_garp(log)
        garp_ms = (time.perf_counter() - t0) * 1000

        # AEI — skip binary search if already consistent
        t0 = time.perf_counter()
        if garp.is_consistent:
            aei_val = 1.0
        else:
            aei_result = compute_aei(log, tolerance=1e-4)
            aei_val = aei_result.efficiency_index
        aei_ms = (time.perf_counter() - t0) * 1000

        # Houtman-Maks — skip if consistent
        t0 = time.perf_counter()
        if garp.is_consistent:
            hm_frac = 0.0
        else:
            hm_result = compute_houtman_maks_index(log)
            hm_frac = hm_result.fraction
        hm_ms = (time.perf_counter() - t0) * 1000

        return UserResult(
            user_id=user_id,
            n_orders=n_orders,
            n_menu_items=n_items,
            is_consistent=garp.is_consistent,
            aei=aei_val,
            hm_fraction=hm_frac,
            garp_time_ms=garp_ms,
            aei_time_ms=aei_ms,
            hm_time_ms=hm_ms,
        )

    except Exception as e:
        return UserResult(
            user_id=user_id,
            n_orders=n_orders,
            n_menu_items=n_items,
            is_consistent=False,
            aei=0.0,
            hm_fraction=1.0,
            garp_time_ms=0.0,
            aei_time_ms=0.0,
            hm_time_ms=0.0,
            error=str(e),
        )


# Module-level wrapper for pickling in ProcessPoolExecutor
def _analyze_user_wrapper(log: BehaviorLog) -> UserResult:
    return analyze_user(log)


# =============================================================================
# Part 3: Parallel Cohort Analysis
# =============================================================================


def analyze_cohort_sequential(logs: list[BehaviorLog]) -> list[UserResult]:
    """Analyze a cohort sequentially (baseline for comparison)."""
    results = []
    for i, log in enumerate(logs):
        results.append(analyze_user(log))
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(logs)} users...")
    return results


def analyze_cohort_parallel(
    logs: list[BehaviorLog],
    max_workers: Optional[int] = None,
) -> list[UserResult]:
    """
    Analyze a cohort of users in parallel using ProcessPoolExecutor.

    Each user's analysis is independent, making this embarrassingly parallel.
    Uses ProcessPoolExecutor to bypass the GIL (important since Numba JIT
    releases the GIL during computation).

    Args:
        logs: List of BehaviorLog objects, one per user
        max_workers: Number of worker processes (None = CPU count)

    Returns:
        List of UserResult objects
    """
    n = len(logs)
    results: list[Optional[UserResult]] = [None] * n
    completed = 0

    workers = max_workers or min(os.cpu_count() or 1, n)
    print(f"    Workers: {workers} processes")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_analyze_user_wrapper, log): i
            for i, log in enumerate(logs)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1
            if completed % 100 == 0 or completed == n:
                print(f"    Completed {completed}/{n} users...")

    return [r for r in results if r is not None]


# =============================================================================
# Part 4: Results Summary
# =============================================================================


def print_cohort_summary(results: list[UserResult], wall_time: float) -> None:
    """Print a comprehensive summary of cohort analysis results."""
    n = len(results)
    errors = [r for r in results if r.error]
    valid = [r for r in results if not r.error]

    aeis = np.array([r.aei for r in valid])
    hms = np.array([r.hm_fraction for r in valid])
    orders = np.array([r.n_orders for r in valid])
    consistent = sum(1 for r in valid if r.is_consistent)

    total_garp_ms = sum(r.garp_time_ms for r in valid)
    total_aei_ms = sum(r.aei_time_ms for r in valid)
    total_hm_ms = sum(r.hm_time_ms for r in valid)

    print(f"\n  Cohort: {n} users ({len(errors)} errors)")
    print(f"  Wall-clock time: {wall_time:.2f}s")
    print(f"  Throughput: {n / wall_time:.0f} users/sec")

    print(f"\n  Order Distribution:")
    print(f"    Min: {int(orders.min()):,}  |  P25: {int(np.percentile(orders, 25)):,}"
          f"  |  Median: {int(np.median(orders)):,}  |  P75: {int(np.percentile(orders, 75)):,}"
          f"  |  Max: {int(orders.max()):,}")

    print(f"\n  GARP Consistency:")
    print(f"    Consistent: {consistent}/{len(valid)} ({consistent/len(valid)*100:.1f}%)")

    print(f"\n  Afriat Efficiency Index (AEI):")
    print(f"    Mean: {aeis.mean():.4f}  |  Std: {aeis.std():.4f}")
    print(f"    P10: {np.percentile(aeis, 10):.3f}  |  P25: {np.percentile(aeis, 25):.3f}"
          f"  |  P50: {np.percentile(aeis, 50):.3f}  |  P75: {np.percentile(aeis, 75):.3f}"
          f"  |  P90: {np.percentile(aeis, 90):.3f}")

    print(f"\n  Houtman-Maks (fraction removed for consistency):")
    hm_nonzero = hms[hms > 0]
    if len(hm_nonzero) > 0:
        print(f"    Mean (inconsistent users): {hm_nonzero.mean():.3f}")
        print(f"    Median: {np.median(hm_nonzero):.3f}  |  Max: {hm_nonzero.max():.3f}")

    print(f"\n  Compute Time Breakdown (total across all users):")
    print(f"    GARP: {total_garp_ms / 1000:.2f}s"
          f"  |  AEI: {total_aei_ms / 1000:.2f}s"
          f"  |  HM: {total_hm_ms / 1000:.2f}s")

    # Scaling projections
    users_per_sec = n / wall_time
    print(f"\n  Scaling Projections (at {users_per_sec:.0f} users/sec):")
    for label, count in [
        ("City cohort (100K)", 100_000),
        ("1% national sample (950K)", 950_000),
        ("Full user base (95M)", 95_000_000),
    ]:
        secs = count / users_per_sec
        if secs < 60:
            time_str = f"{secs:.0f}s"
        elif secs < 3600:
            time_str = f"{secs / 60:.1f} min"
        else:
            time_str = f"{secs / 3600:.1f} hours"
        # Linear scaling with more cores
        time_64 = secs * (max(1, os.cpu_count() or 1)) / 64
        if time_64 < 60:
            time_64_str = f"{time_64:.0f}s"
        elif time_64 < 3600:
            time_64_str = f"{time_64 / 60:.1f} min"
        else:
            time_64_str = f"{time_64 / 3600:.1f} hours"
        print(f"    {label}: {time_str} (this machine) | ~{time_64_str} (64-core server)")


# =============================================================================
# Part 5: Main Demo
# =============================================================================


def print_banner(text: str, char: str = "=", width: int = 70) -> None:
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uber Eats Revealed Preference Analysis at Scale"
    )
    parser.add_argument("--users", type=int, default=100,
                        help="Number of users in cohort (default: 100)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: CPU count)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially instead of parallel")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    print_banner("UBER EATS: REVEALED PREFERENCE AT SCALE")
    print(" Simulating food-delivery order data and analyzing behavioral consistency")
    print(" Uses SCC-optimized GARP + AEI + Houtman-Maks with parallel processing")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Part A: Single-User Benchmarks
    # -----------------------------------------------------------------
    print_banner("[1/3] SINGLE-USER BENCHMARKS", "-", 60)
    print("  Analyzing individual users at different order volumes...\n")

    for label, n_orders, n_menu in [
        ("Casual (3 months)", 30, 60),
        ("Regular (6 months)", 100, 100),
        ("Active (1 year)", 365, 120),
        ("Power user (2 years)", 700, 150),
    ]:
        log = simulate_uber_eats_user(
            user_id=label, n_orders=n_orders,
            n_menu_items=n_menu, seed=args.seed,
        )
        t0 = time.perf_counter()
        result = analyze_user(log)
        wall = (time.perf_counter() - t0) * 1000

        status = "[+] consistent" if result.is_consistent else f"[-] AEI={result.aei:.3f}"
        print(f"  {label:25s}  T={n_orders:4d}  N={n_menu:3d}  "
              f"{status:20s}  HM={result.hm_fraction:.3f}  "
              f"time={wall:.0f}ms")

    # -----------------------------------------------------------------
    # Part B: Parallel Cohort Analysis
    # -----------------------------------------------------------------
    print_banner(f"[2/3] COHORT ANALYSIS ({args.users} USERS)", "-", 60)

    print(f"  Simulating {args.users} heterogeneous users...")
    t0 = time.perf_counter()
    logs = simulate_uber_eats_cohort(
        n_users=args.users, n_menu_items=100, seed=args.seed,
    )
    sim_time = time.perf_counter() - t0
    print(f"  Simulation: {sim_time:.2f}s")

    # Print archetype breakdown
    order_counts = [log.num_records for log in logs]
    print(f"\n  User heterogeneity:")
    brackets = [(0, 20, "Churned"), (20, 60, "Casual"), (60, 200, "Regular"),
                (200, 500, "Power"), (500, 10000, "Super-power")]
    for lo, hi, label in brackets:
        count = sum(1 for o in order_counts if lo <= o < hi)
        if count > 0:
            print(f"    {label:15s}: {count:4d} users  "
                  f"({lo}-{min(hi, max(order_counts))} orders)")

    print(f"\n  Running analysis {'sequentially' if args.sequential else 'in parallel'}...")
    t0 = time.perf_counter()
    if args.sequential:
        results = analyze_cohort_sequential(logs)
    else:
        results = analyze_cohort_parallel(logs, max_workers=args.workers)
    wall_time = time.perf_counter() - t0

    print_cohort_summary(results, wall_time)

    # -----------------------------------------------------------------
    # Part C: Scaling Projections
    # -----------------------------------------------------------------
    print_banner("[3/3] PRODUCTION DEPLOYMENT RECOMMENDATIONS", "-", 60)

    print("""
  For a production deployment analyzing Uber Eats user consistency:

  USE CASES:
    - Bot detection: AEI < 0.5 flags non-human ordering patterns
    - User segmentation: AEI distribution reveals behavioral clusters
    - UX evaluation: A/B test whether UI changes improve choice consistency
    - Fraud detection: Sudden AEI drops may indicate account sharing

  ARCHITECTURE:
    - Per-user analysis is embarrassingly parallel (no shared state)
    - Each user takes 10-500ms depending on order history length
    - ProcessPoolExecutor scales linearly with cores
    - For cloud: map each user to a Lambda/Cloud Function invocation

  BATCH SCHEDULE:
    - Nightly batch: analyze all users who ordered today
    - Weekly full sweep: re-analyze all active users
    - On-demand: trigger on suspicious activity flags
""")

    print_banner("DONE", "=", 70)


if __name__ == "__main__":
    main()
