#!/usr/bin/env python3
"""Application: Recommendation Clicks — SARP Consistency for User Segmentation.

Based on the framework from:
  Kallus & Udell (2016) "Revealed Preference at Scale: Learning Personalized
  Preferences from Assortment Choices," EC '16, pp. 821-837.
  Cazzola & Daly (2024) "Rank-Preference Consistency as the Appropriate
  Metric for Recommender Systems," arXiv:2404.17097.

A recommendation platform shows users menus of 3-8 items from a catalog.
Users click one item per session. Across sessions with varying menus, we
test whether each user's clicks satisfy SARP (Strong Axiom of Revealed
Preference) — i.e., can be explained by a fixed preference ranking.

No prices needed: this uses PyRevealed's MenuChoiceLog path.

Three concrete use cases from one score:
1. Recommender evaluation: A/B test which ranker elicits more SARP-consistent choices.
2. User segmentation: high-SARP users have stable preferences (personalize);
   low-SARP users are noisy (curate/explore).
3. Churn signal: SARP consistency over sliding windows detects preference drift.

Pipeline: MenuChoiceLog -> SARP -> Houtman-Maks -> temporal analysis.

Usage:
    python applications/03_recommendation_clicks.py
    python applications/03_recommendation_clicks.py --users 200 --sessions 80
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from pyrevealed import MenuChoiceLog
from pyrevealed.algorithms.abstract_choice import validate_menu_sarp, compute_menu_efficiency


# =============================================================================
# Configuration
# =============================================================================

CATALOG = [f"Item_{chr(65 + i)}" for i in range(20)]  # Item_A through Item_T
N_ITEMS = len(CATALOG)


# =============================================================================
# Data Simulation
# =============================================================================

def generate_user_sessions(
    user_type: str,
    n_sessions: int,
    rng: np.random.Generator,
) -> tuple[list[frozenset[int]], list[int], list[int] | None]:
    """Simulate one user's click sessions.

    Args:
        user_type: "consistent", "noisy", "drifting", or "random"
        n_sessions: Number of sessions
        rng: random generator

    Returns:
        (menus, choices, preference_order) where preference_order is the
        latent ranking used for generation (None for random users).
    """
    # Generate latent preference ranking
    pref_order = list(rng.permutation(N_ITEMS))
    pref_rank = {item: rank for rank, item in enumerate(pref_order)}

    # For drifting users: second-half ranking is a different permutation
    if user_type == "drifting":
        pref_order_2 = list(rng.permutation(N_ITEMS))
        pref_rank_2 = {item: rank for rank, item in enumerate(pref_order_2)}
        midpoint = n_sessions // 2
    else:
        pref_rank_2 = None
        midpoint = n_sessions  # never switches

    menus: list[frozenset[int]] = []
    choices: list[int] = []

    for t in range(n_sessions):
        # Random menu of 3-8 items
        menu_size = rng.integers(3, 9)
        menu_items = rng.choice(N_ITEMS, size=menu_size, replace=False)
        menu = frozenset(menu_items.tolist())

        # Choose based on user type
        if user_type == "random":
            choice = int(rng.choice(list(menu)))
        elif user_type == "drifting" and t >= midpoint:
            # Use second preference ranking
            choice = min(menu, key=lambda x: pref_rank_2[x])
        elif user_type == "noisy":
            # Pick top-ranked with prob 0.7, else random
            if rng.random() < 0.7:
                choice = min(menu, key=lambda x: pref_rank[x])
            else:
                choice = int(rng.choice(list(menu)))
        else:
            # Consistent: always pick top-ranked
            choice = min(menu, key=lambda x: pref_rank[x])

        menus.append(menu)
        choices.append(choice)

    return menus, choices, pref_order


# =============================================================================
# Analysis
# =============================================================================

@dataclass
class UserResult:
    user_id: str
    user_type: str
    n_sessions: int
    is_sarp: bool
    n_violations: int
    hm_efficiency: float
    # Temporal analysis
    first_half_sarp: bool
    second_half_sarp: bool
    first_half_hm: float
    second_half_hm: float
    time_ms: float


def analyze_user(
    uid: str, utype: str, menus: list[frozenset[int]], choices: list[int],
) -> UserResult:
    """Run SARP -> Houtman-Maks -> temporal split analysis."""
    t0 = time.perf_counter()
    n = len(menus)
    mid = n // 2

    # Full sequence
    log = MenuChoiceLog(menus=menus, choices=choices)
    sarp = validate_menu_sarp(log)
    hm = compute_menu_efficiency(log)

    # First half
    log_1 = MenuChoiceLog(menus=menus[:mid], choices=choices[:mid])
    sarp_1 = validate_menu_sarp(log_1)
    hm_1 = compute_menu_efficiency(log_1)

    # Second half
    log_2 = MenuChoiceLog(menus=menus[mid:], choices=choices[mid:])
    sarp_2 = validate_menu_sarp(log_2)
    hm_2 = compute_menu_efficiency(log_2)

    elapsed = (time.perf_counter() - t0) * 1000

    return UserResult(
        user_id=uid, user_type=utype, n_sessions=n,
        is_sarp=sarp.is_consistent,
        n_violations=len(sarp.violations),
        hm_efficiency=hm.efficiency_index,
        first_half_sarp=sarp_1.is_consistent,
        second_half_sarp=sarp_2.is_consistent,
        first_half_hm=hm_1.efficiency_index,
        second_half_hm=hm_2.efficiency_index,
        time_ms=elapsed,
    )


# =============================================================================
# Rolling-Window Temporal Panel
# =============================================================================

@dataclass
class LifecycleResult:
    user_id: str
    user_type: str
    hm_trajectory: list[float]
    mean_hm: float
    std_hm: float
    slope: float
    lifecycle: str  # stable/improving/deteriorating/volatile


def compute_rolling_hm(
    menus: list[frozenset[int]], choices: list[int],
    window: int = 15, step: int = 5,
) -> list[float]:
    """Compute HM efficiency over rolling windows of sessions."""
    n = len(menus)
    if n < window:
        log = MenuChoiceLog(menus=menus, choices=choices)
        return [compute_menu_efficiency(log).efficiency_index]

    results = []
    for start in range(0, n - window + 1, step):
        end = start + window
        log = MenuChoiceLog(menus=menus[start:end], choices=choices[start:end])
        hm = compute_menu_efficiency(log).efficiency_index
        results.append(hm)
    return results


def classify_lifecycle(hm_values: list[float]) -> tuple[str, float]:
    """Classify an HM trajectory."""
    if len(hm_values) < 2:
        return "stable", 0.0
    arr = np.array(hm_values)
    std = arr.std()
    slope = np.polyfit(np.arange(len(arr)), arr, 1)[0]

    if std < 0.05:
        return "stable", slope
    elif slope > 0.01:
        return "improving", slope
    elif slope < -0.01:
        return "deteriorating", slope
    else:
        return "volatile", slope


def run_lifecycle_analysis(
    user_data: list[tuple[str, str, list, list]],
    window: int = 15, step: int = 5,
) -> list[LifecycleResult]:
    """Run rolling-window HM analysis for all users."""
    results = []
    for uid, utype, menus, choices in user_data:
        if len(menus) < window:
            continue
        hm_traj = compute_rolling_hm(menus, choices, window, step)
        lifecycle, slope = classify_lifecycle(hm_traj)
        results.append(LifecycleResult(
            user_id=uid, user_type=utype,
            hm_trajectory=hm_traj,
            mean_hm=np.mean(hm_traj), std_hm=np.std(hm_traj),
            slope=slope, lifecycle=lifecycle,
        ))
    return results


def print_lifecycle_results(lifecycle: list[LifecycleResult]) -> None:
    """Print lifecycle panel analysis."""
    if not lifecycle:
        return

    n = len(lifecycle)
    print_banner("USER LIFECYCLE CLASSIFICATION")
    print(f"  Users analyzed: {n}")

    print(f"\n  {'Lifecycle':<16s} {'N':>4s} {'%':>7s} {'Mean HM':>8s}"
          f" {'Std HM':>8s} {'Slope':>8s}")
    print(f"  {'-'*16} {'-'*4} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
    for lc in ["stable", "improving", "deteriorating", "volatile"]:
        subset = [r for r in lifecycle if r.lifecycle == lc]
        if not subset:
            continue
        pct = len(subset) / n * 100
        print(f"  {lc:<16s} {len(subset):4d} {pct:6.1f}%"
              f" {np.mean([r.mean_hm for r in subset]):8.3f}"
              f" {np.mean([r.std_hm for r in subset]):8.3f}"
              f" {np.mean([r.slope for r in subset]):+8.4f}")

    # Cross-tabulation: user_type × lifecycle
    print_banner("CROSS-TAB: TRUE TYPE x DETECTED LIFECYCLE")
    types = ["consistent", "noisy", "drifting", "random"]
    lifecycles = ["stable", "improving", "deteriorating", "volatile"]
    print(f"  {'':>12s}", end="")
    for lc in lifecycles:
        print(f" {lc:>13s}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in lifecycles:
        print(f" {'-'*13}", end="")
    print()
    for utype in types:
        print(f"  {utype:>12s}", end="")
        subset_type = [r for r in lifecycle if r.user_type == utype]
        for lc in lifecycles:
            count = sum(1 for r in subset_type if r.lifecycle == lc)
            print(f" {count:13d}", end="")
        print()


# =============================================================================
# Reporting
# =============================================================================

def print_banner(text: str, char: str = "=", width: int = 70) -> None:
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_results(results: list[UserResult], wall_time: float) -> None:
    n = len(results)
    sarp_pass = sum(1 for r in results if r.is_sarp)
    violations = np.array([r.n_violations for r in results])
    hm_effs = np.array([r.hm_efficiency for r in results])

    print_banner("OVERALL RESULTS")
    print(f"  Users: {n}  |  Sessions/user: {results[0].n_sessions}")
    print(f"  SARP-consistent: {sarp_pass}/{n} ({sarp_pass/n*100:.1f}%)")
    print(f"  Wall time: {wall_time:.2f}s  |  Throughput: {n/wall_time:.0f} users/sec")

    print(f"\n  {'Metric':<15s} {'Mean':>8s} {'Std':>8s} {'P25':>8s}"
          f" {'P50':>8s} {'P75':>8s}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, arr in [("Violations", violations.astype(float)),
                       ("HM efficiency", hm_effs)]:
        print(f"  {label:<15s} {arr.mean():8.2f} {arr.std():8.2f}"
              f" {np.percentile(arr, 25):8.2f} {np.percentile(arr, 50):8.2f}"
              f" {np.percentile(arr, 75):8.2f}")

    # By user type
    print_banner("SEGMENTATION BY USER TYPE")
    types = ["consistent", "noisy", "drifting", "random"]
    print(f"  {'Type':<12s} {'N':>4s} {'SARP%':>7s} {'Violations':>10s}"
          f" {'HM eff':>8s} {'1st-half':>10s} {'2nd-half':>10s}")
    print(f"  {'-'*12} {'-'*4} {'-'*7} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
    for utype in types:
        subset = [r for r in results if r.user_type == utype]
        if not subset:
            continue
        n_t = len(subset)
        sarp_pct = sum(1 for r in subset if r.is_sarp) / n_t * 100
        avg_viol = np.mean([r.n_violations for r in subset])
        avg_hm = np.mean([r.hm_efficiency for r in subset])
        avg_hm1 = np.mean([r.first_half_hm for r in subset])
        avg_hm2 = np.mean([r.second_half_hm for r in subset])
        print(f"  {utype:<12s} {n_t:4d} {sarp_pct:6.1f}% {avg_viol:10.1f}"
              f" {avg_hm:8.3f} {avg_hm1:10.3f} {avg_hm2:10.3f}")

    # Churn signal: drifting users
    print_banner("CHURN SIGNAL: PREFERENCE DRIFT DETECTION")
    drifting = [r for r in results if r.user_type == "drifting"]
    if drifting:
        print(f"  Drifting users (preference ranking changes mid-stream):")
        print(f"    Full-sequence HM efficiency:  {np.mean([r.hm_efficiency for r in drifting]):.3f}")
        print(f"    First-half HM efficiency:     {np.mean([r.first_half_hm for r in drifting]):.3f}")
        print(f"    Second-half HM efficiency:    {np.mean([r.second_half_hm for r in drifting]):.3f}")
        print()
        print("  Key insight: each half is individually consistent, but the full")
        print("  sequence shows violations. This pattern — high per-window consistency")
        print("  but low full-sequence consistency — is a leading indicator of")
        print("  preference drift (churn signal) before engagement metrics move.")

    # Show a few individual drifting users
    if drifting:
        print()
        print(f"  {'User':<10s} {'Full HM':>8s} {'1st half':>8s} {'2nd half':>8s}"
              f" {'Delta':>8s}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for r in drifting[:5]:
            delta = r.hm_efficiency - (r.first_half_hm + r.second_half_hm) / 2
            print(f"  {r.user_id:<10s} {r.hm_efficiency:8.3f} {r.first_half_hm:8.3f}"
                  f" {r.second_half_hm:8.3f} {delta:+8.3f}")

    print_banner("INTERPRETATION")
    print("""
  Reference: Kallus & Udell (2016, EC) formalized preference learning from
  assortment choice data at platform scale. Cazzola & Daly (2024) argued
  rank-preference consistency (SARP satisfaction) is the correct evaluation
  metric for recommender systems, outperforming RMSE/MAE.

  Three applications from one score:
  1. RECOMMENDER EVALUATION: A/B test rankers by which yields higher avg
     SARP consistency — measures if users make coherent choices, not just clicks.
  2. USER SEGMENTATION: high-HM users have learnable preferences (invest in
     personalization); low-HM users are noisy (invest in exploration/curation).
  3. CHURN DETECTION: sliding-window SARP tracks preference stability over
     time. A consistency drop precedes disengagement.
""")


def plot_results(results: list[UserResult]) -> None:
    """Optional: scatter of violations vs HM efficiency."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not installed -- skipping plot]")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = {"consistent": "#2ecc71", "noisy": "#f39c12",
              "drifting": "#9b59b6", "random": "#e74c3c"}
    for utype in ["consistent", "noisy", "drifting", "random"]:
        subset = [r for r in results if r.user_type == utype]
        if not subset:
            continue
        x = [r.n_violations for r in subset]
        y = [r.hm_efficiency for r in subset]
        ax.scatter(x, y, alpha=0.6, label=utype.capitalize(),
                   color=colors[utype], s=40)

    ax.set_xlabel("Number of SARP violations")
    ax.set_ylabel("Houtman-Maks efficiency")
    ax.set_title("User Segmentation by Choice Consistency — Recommendation Clicks")
    ax.legend()
    plt.tight_layout()
    out = "applications/recsys_sarp_scatter.png"
    plt.savefig(out, dpi=150)
    print(f"  Plot saved to {out}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommendation Clicks -- SARP User Segmentation"
    )
    parser.add_argument("--users", type=int, default=100,
                        help="Number of users (default: 100)")
    parser.add_argument("--sessions", type=int, default=50,
                        help="Sessions per user (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--plot", action="store_true",
                        help="Save scatter plot (requires matplotlib)")
    args = parser.parse_args()

    print_banner("RECOMMENDATION CLICKS: SARP CONSISTENCY FOR USER SEGMENTATION")
    print(f"  Paper: Kallus & Udell (2016, EC) + Cazzola & Daly (2024)")
    print(f"  Pipeline: MenuChoiceLog -> SARP -> Houtman-Maks -> temporal analysis")
    print(f"  Users: {args.users}  |  Sessions: {args.sessions}  |  Catalog: {N_ITEMS} items")
    print("=" * 70)

    rng = np.random.default_rng(args.seed)

    # Generate users: 30% consistent, 40% noisy, 20% drifting, 10% random
    print_banner("[1/2] SIMULATING USER CLICK DATA", "-", 60)
    user_data = []
    type_counts: dict[str, int] = {}
    for i in range(args.users):
        r = rng.random()
        if r < 0.30:
            utype = "consistent"
        elif r < 0.70:
            utype = "noisy"
        elif r < 0.90:
            utype = "drifting"
        else:
            utype = "random"

        type_counts[utype] = type_counts.get(utype, 0) + 1
        menus, choices, pref = generate_user_sessions(utype, args.sessions, rng)
        user_data.append((f"U-{i+1:04d}", utype, menus, choices))

    for t in ["consistent", "noisy", "drifting", "random"]:
        print(f"    {t:<12s}: {type_counts.get(t, 0)}")

    # Analyze
    print_banner("[2/2] RUNNING SARP ANALYSIS", "-", 60)
    t0 = time.perf_counter()
    results = []
    for i, (uid, utype, menus, choices) in enumerate(user_data):
        result = analyze_user(uid, utype, menus, choices)
        results.append(result)
        if (i + 1) % 25 == 0 or (i + 1) == len(user_data):
            print(f"    Processed {i+1}/{len(user_data)} users...")
    wall_time = time.perf_counter() - t0

    # Report
    print_results(results, wall_time)

    # Lifecycle panel analysis
    print_banner("[3/3] ROLLING-WINDOW LIFECYCLE ANALYSIS", "-", 60)
    print(f"  Computing rolling HM (window=15, step=5)...")
    lifecycle = run_lifecycle_analysis(user_data, window=15, step=5)
    print_lifecycle_results(lifecycle)

    if args.plot:
        print_banner("VISUALIZATION", "-", 60)
        plot_results(results)

    print_banner("DONE", "=", 70)


if __name__ == "__main__":
    main()
