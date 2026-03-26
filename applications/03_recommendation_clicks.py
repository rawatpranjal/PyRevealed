#!/usr/bin/env python3
"""Application: Recommendation Clicks — SARP from RetailRocket Click-Stream.

Based on:
  Kallus & Udell (2016) "Revealed Preference at Scale," EC '16.
  Cazzola & Daly (2024) arXiv:2404.17097.

Loads RetailRocket e-commerce click-stream data, reconstructs menu-choice
observations (items viewed in a session = menu, purchased item = choice),
and batch-scores SARP consistency via the Rust Engine. Includes rolling-
window lifecycle analysis for churn detection.

Data: RetailRocket (Kaggle, CC-BY-NC-SA 4.0). 2.75M events, 1.4M visitors.
      Download: kaggle datasets download -d retailrocket/ecommerce-dataset

Usage:
    python applications/03_recommendation_clicks.py
    python applications/03_recommendation_clicks.py --max-users 500
    python applications/03_recommendation_clicks.py --plot
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from pyrevealed import MenuChoiceLog
from pyrevealed.algorithms.abstract_choice import compute_menu_efficiency
from pyrevealed.datasets import load_retailrocket
from pyrevealed.engine import Engine


# =============================================================================
# Batch Analysis (Rust Engine)
# =============================================================================

@dataclass
class UserResult:
    user_id: str
    n_sessions: int
    n_items: int
    is_sarp: bool
    n_violations: int
    hm_efficiency: float
    first_half_hm: float
    second_half_hm: float
    time_us: int


def run_engine_batch(
    user_logs: dict[str, MenuChoiceLog],
) -> list[UserResult]:
    """Score all users using the Rust Engine for maximum throughput."""
    engine = Engine()
    uids = list(user_logs.keys())
    logs = list(user_logs.values())

    # --- Full-data batch ---
    full_tuples = [log.to_engine_tuple() for log in logs]
    print(f"  Engine backend: {engine.backend}")
    print(f"  Scoring {len(full_tuples)} users...")
    t0 = time.perf_counter()
    full_results = engine.analyze_menus(full_tuples)

    # --- Split-half batches ---
    # Build first-half and second-half tuples for all users with enough data
    fh_tuples: list[tuple[list[list[int]], list[int], int]] = []
    sh_tuples: list[tuple[list[list[int]], list[int], int]] = []
    split_mask: list[bool] = []  # which users have valid splits

    for log in logs:
        n = len(log.choices)
        mid = n // 2
        if mid >= 3:
            split_mask.append(True)
            fh_log = MenuChoiceLog(menus=log.menus[:mid], choices=log.choices[:mid])
            sh_log = MenuChoiceLog(menus=log.menus[mid:], choices=log.choices[mid:])
            fh_tuples.append(fh_log.to_engine_tuple())
            sh_tuples.append(sh_log.to_engine_tuple())
        else:
            split_mask.append(False)

    fh_results = engine.analyze_menus(fh_tuples) if fh_tuples else []
    sh_results = engine.analyze_menus(sh_tuples) if sh_tuples else []
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.2f}s ({len(uids)/elapsed:.0f} users/sec)")

    # --- Assemble UserResult objects ---
    results: list[UserResult] = []
    split_idx = 0
    for i, (uid, log, mr) in enumerate(zip(uids, logs, full_results)):
        hm_eff = mr.hm_consistent / max(mr.hm_total, 1)
        n_items = len(set().union(*log.menus))

        if split_mask[i]:
            fh_hm = fh_results[split_idx].hm_consistent / max(fh_results[split_idx].hm_total, 1)
            sh_hm = sh_results[split_idx].hm_consistent / max(sh_results[split_idx].hm_total, 1)
            split_idx += 1
        else:
            fh_hm = sh_hm = hm_eff

        results.append(UserResult(
            user_id=uid, n_sessions=len(log.choices), n_items=n_items,
            is_sarp=mr.is_sarp,
            n_violations=mr.n_sarp_violations,
            hm_efficiency=hm_eff,
            first_half_hm=fh_hm, second_half_hm=sh_hm,
            time_us=mr.compute_time_us,
        ))
    return results


# =============================================================================
# Rolling-Window Lifecycle (per-user window slicing — inherently sequential)
# =============================================================================

@dataclass
class LifecycleResult:
    user_id: str
    hm_trajectory: list[float]
    mean_hm: float
    std_hm: float
    slope: float
    lifecycle: str


def compute_rolling_hm(log: MenuChoiceLog, window: int = 10, step: int = 3) -> list[float]:
    """Compute HM efficiency over rolling session windows."""
    n = len(log.choices)
    if n < window:
        return [compute_menu_efficiency(log).efficiency_index]

    results = []
    for start in range(0, n - window + 1, step):
        end = start + window
        window_log = MenuChoiceLog(menus=log.menus[start:end], choices=log.choices[start:end])
        hm = compute_menu_efficiency(window_log).efficiency_index
        results.append(hm)
    return results


def classify_lifecycle(hm_values: list[float]) -> tuple[str, float]:
    if len(hm_values) < 2:
        return "stable", 0.0
    arr = np.array(hm_values)
    slope = np.polyfit(np.arange(len(arr)), arr, 1)[0]
    std = arr.std()
    if std < 0.05:
        return "stable", slope
    elif slope > 0.01:
        return "improving", slope
    elif slope < -0.01:
        return "deteriorating", slope
    else:
        return "volatile", slope


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
    sessions = np.array([r.n_sessions for r in results])
    items = np.array([r.n_items for r in results])

    print_banner("DATASET SUMMARY")
    print(f"  Data: RetailRocket e-commerce click-stream")
    print(f"  Users: {n}  |  Total sessions: {sessions.sum():,}")
    print(f"  Sessions/user:  mean={sessions.mean():.1f}  median={np.median(sessions):.0f}"
          f"  max={sessions.max()}")
    print(f"  Items/user:     mean={items.mean():.1f}  median={np.median(items):.0f}"
          f"  max={items.max()}")

    print_banner("SARP CONSISTENCY")
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

    # Drift detection
    print_banner("DRIFT DETECTION (SPLIT-HALF)")
    drifters = []
    for r in results:
        delta = r.hm_efficiency - (r.first_half_hm + r.second_half_hm) / 2
        if abs(delta) > 0.1:
            drifters.append((r, delta))
    drifters.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"  Users with significant drift (|delta| > 0.1): {len(drifters)}/{n}")
    if drifters:
        print(f"\n  {'User':<12s} {'Sessions':>8s} {'Full HM':>8s} {'1st half':>8s}"
              f" {'2nd half':>8s} {'Delta':>8s}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for r, delta in drifters[:10]:
            print(f"  {r.user_id:<12s} {r.n_sessions:8d} {r.hm_efficiency:8.3f}"
                  f" {r.first_half_hm:8.3f} {r.second_half_hm:8.3f} {delta:+8.3f}")


def print_lifecycle_results(lifecycle: list[LifecycleResult]) -> None:
    if not lifecycle:
        return
    n = len(lifecycle)

    print_banner("USER LIFECYCLE CLASSIFICATION")
    print(f"  Users with 10+ sessions: {n}")

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


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommendation Clicks — RetailRocket SARP Analysis"
    )
    parser.add_argument("--max-users", type=int, default=None,
                        help="Max users to analyze (default: all)")
    parser.add_argument("--min-sessions", type=int, default=5,
                        help="Min sessions per user (default: 5)")
    parser.add_argument("--plot", action="store_true",
                        help="Save scatter plot (requires matplotlib)")
    args = parser.parse_args()

    print_banner("RECOMMENDATION CLICKS: RETAILROCKET SARP PANEL")
    print(f"  Papers: Kallus & Udell (2016, EC) + Cazzola & Daly (2024)")
    print(f"  Data: RetailRocket e-commerce click-stream (Kaggle)")
    print(f"  Pipeline: Rust Engine → SARP/WARP/HM batch → lifecycle")
    print("=" * 70)

    # Load real data
    print_banner("[1/3] LOADING RETAILROCKET DATA", "-", 60)
    user_logs = load_retailrocket(
        min_sessions=args.min_sessions,
        max_users=args.max_users,
    )
    print(f"  Loaded {len(user_logs)} users")

    # Batch score with Rust Engine
    print_banner("[2/3] BATCH SCORING (RUST ENGINE)", "-", 60)
    t0 = time.perf_counter()
    results = run_engine_batch(user_logs)
    wall_time = time.perf_counter() - t0
    print_results(results, wall_time)

    # Lifecycle analysis (per-user window slicing)
    print_banner("[3/3] ROLLING-WINDOW LIFECYCLE", "-", 60)
    lifecycle = []
    for uid, log in user_logs.items():
        if len(log.choices) < 10:
            continue
        hm_traj = compute_rolling_hm(log, window=10, step=3)
        lc, slope = classify_lifecycle(hm_traj)
        lifecycle.append(LifecycleResult(
            user_id=uid, hm_trajectory=hm_traj,
            mean_hm=np.mean(hm_traj), std_hm=np.std(hm_traj),
            slope=slope, lifecycle=lc,
        ))
    print_lifecycle_results(lifecycle)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            x = [r.n_violations for r in results]
            y = [r.hm_efficiency for r in results]
            ax.scatter(x, y, alpha=0.5, s=20, color="#3498db")
            ax.set_xlabel("SARP violations")
            ax.set_ylabel("Houtman-Maks efficiency")
            ax.set_title(f"RetailRocket: {len(results)} Users")
            plt.tight_layout()
            plt.savefig("applications/recsys_sarp_scatter.png", dpi=150)
            print(f"\n  Plot saved to applications/recsys_sarp_scatter.png")
            plt.close()
        except ImportError:
            print("  [matplotlib not installed]")

    print_banner("DONE", "=", 70)


if __name__ == "__main__":
    main()
