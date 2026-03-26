#!/usr/bin/env python3
"""Application: Grocery Scanner Panel Data — Rationality Scoring at Scale.

Replicates the revealed preference analysis from:
  Dean & Martin (2016) "Measuring Rationality with the Minimum Cost of
  Revealed Preference Violations," AER 106(11), 3297-3329.
  Echenique, Lee & Shum (2011) "The Money Pump as a Measure of Revealed
  Preference Violations," JPE 119(6), 1201-1223.

Scores 2,222 households from the Dunnhumby "The Complete Journey" dataset
using the Rust engine for batch GARP/CCEI/MPI/HM computation, then runs
rolling-window CCEI to classify household trajectories over time.

Data: Dunnhumby loyalty-card scanner data (10 product categories, 104 weeks).
      Download: dunnhumby/download_data.sh

Usage:
    python applications/01_grocery_scanner.py
    python applications/01_grocery_scanner.py --households 500
    python applications/01_grocery_scanner.py --plot
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyrevealed import BehaviorLog, validate_consistency, compute_integrity_score
from pyrevealed.engine import Engine


# =============================================================================
# Configuration
# =============================================================================

REPO_ROOT = Path(__file__).parent.parent
DUNNHUMBY_DIR = REPO_ROOT / "dunnhumby"
TRANSACTION_FILE = DUNNHUMBY_DIR / "data" / "transaction_data.csv"


# =============================================================================
# Data Loading (Dunnhumby only)
# =============================================================================

def load_dunnhumby(max_households: int | None = None) -> list[tuple[str, BehaviorLog]]:
    """Load real Dunnhumby grocery data.

    Returns list of (household_id, BehaviorLog).

    Raises:
        FileNotFoundError: If Dunnhumby data hasn't been downloaded.
    """
    if not TRANSACTION_FILE.exists():
        raise FileNotFoundError(
            f"Dunnhumby data not found at {TRANSACTION_FILE}\n\n"
            "Download it first:\n"
            "  cd dunnhumby && bash download_data.sh\n\n"
            "This requires a Kaggle account. See dunnhumby/README.md for details."
        )

    sys.path.insert(0, str(DUNNHUMBY_DIR))
    try:
        from data_loader import load_filtered_data
        from price_oracle import get_master_price_grid
        from session_builder import build_all_sessions
    finally:
        sys.path.pop(0)

    print("  Loading Dunnhumby transaction data...")
    filtered = load_filtered_data(use_cache=True)
    price_grid = get_master_price_grid(filtered, use_cache=True)

    print("  Building household sessions...")
    households_dict = build_all_sessions(filtered, price_grid)

    households = []
    for hh_key, hh_data in households_dict.items():
        households.append((f"HH-{hh_key}", hh_data.behavior_log))
        if max_households and len(households) >= max_households:
            break

    return households


# =============================================================================
# Batch Analysis (Rust Engine)
# =============================================================================

def run_engine_batch(
    household_logs: list[tuple[str, BehaviorLog]],
) -> list[dict]:
    """Score all households using the Rust Engine for maximum throughput."""
    engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])

    # Prepare arrays for Engine
    users = [(log.cost_vectors, log.action_vectors) for _, log in household_logs]

    print(f"  Engine backend: {engine.backend}")
    print(f"  Scoring {len(users)} households...")
    t0 = time.perf_counter()
    engine_results = engine.analyze_arrays(users)
    elapsed = time.perf_counter() - t0

    print(f"  Done in {elapsed:.2f}s ({len(users)/elapsed:.0f} households/sec)")

    results = []
    for (hid, log), er in zip(household_logs, engine_results):
        hm_frac = 1.0 - (er.hm_consistent / er.hm_total) if er.hm_total > 0 else 0.0
        results.append({
            "hid": hid, "T": log.num_records,
            "garp": er.is_garp, "ccei": er.ccei,
            "mpi": er.mpi, "hm_removed": hm_frac,
        })
    return results


# =============================================================================
# Temporal Panel Analysis
# =============================================================================

@dataclass
class TemporalResult:
    household_id: str
    ccei_trajectory: list[float]
    mean_ccei: float
    std_ccei: float
    slope: float
    trajectory_type: str  # stable/improving/deteriorating/volatile


def compute_rolling_ccei(
    log: BehaviorLog, window: int = 20, step: int = 5,
) -> list[float]:
    """Compute CCEI over rolling windows."""
    T = log.num_records
    if T < window:
        return [compute_integrity_score(log, tolerance=1e-4).efficiency_index]

    results = []
    for start in range(0, T - window + 1, step):
        end = start + window
        window_log = BehaviorLog(
            cost_vectors=log.cost_vectors[start:end],
            action_vectors=log.action_vectors[start:end],
        )
        garp = validate_consistency(window_log)
        if garp.is_consistent:
            ccei = 1.0
        else:
            ccei = compute_integrity_score(window_log, tolerance=1e-4).efficiency_index
        results.append(ccei)
    return results


def classify_trajectory(ccei_values: list[float]) -> tuple[str, float]:
    """Classify a CCEI trajectory. Returns (type, slope)."""
    if len(ccei_values) < 2:
        return "stable", 0.0
    arr = np.array(ccei_values)
    slope = np.polyfit(np.arange(len(arr)), arr, 1)[0]
    std = arr.std()
    if std < 0.03:
        return "stable", slope
    elif slope > 0.005:
        return "improving", slope
    elif slope < -0.005:
        return "deteriorating", slope
    else:
        return "volatile", slope


def _compute_cohort_mean(
    all_trajectories: list[list[float]],
) -> dict[int, float]:
    """Compute mean CCEI at each window index across all households.

    This serves as a time fixed effect: if all households' CCEI drops
    at window index 10, that's a macro price effect, not individual
    deterioration.
    """
    by_index: dict[int, list[float]] = {}
    for traj in all_trajectories:
        for i, val in enumerate(traj):
            by_index.setdefault(i, []).append(val)
    return {i: np.mean(vals) for i, vals in by_index.items()}


def run_temporal_analysis(
    household_logs: list[tuple[str, BehaviorLog]],
    window: int = 20, step: int = 5,
) -> list[TemporalResult]:
    """Rolling-window CCEI for all households with enough data.

    Uses a two-pass approach: first compute all raw trajectories, then
    compute the cohort mean CCEI at each window index (time fixed effect),
    then classify each household by its *deviation* from the cohort mean.
    This separates household-level behavioral change from macro price
    environment effects.
    """
    # Pass 1: compute raw trajectories
    raw: list[tuple[str, list[float]]] = []
    for i, (hid, log) in enumerate(household_logs):
        if log.num_records < window:
            continue
        ccei_values = compute_rolling_ccei(log, window, step)
        raw.append((hid, ccei_values))
        if (i + 1) % 100 == 0:
            print(f"    Temporal: {i+1}/{len(household_logs)}...")

    # Pass 2: cohort mean at each window index (time fixed effect)
    cohort_mean = _compute_cohort_mean([traj for _, traj in raw])

    # Pass 3: classify by deviation from cohort mean
    results = []
    for hid, ccei_values in raw:
        deviations = [
            val - cohort_mean.get(i, val) for i, val in enumerate(ccei_values)
        ]
        ttype, slope = classify_trajectory(deviations)
        results.append(TemporalResult(
            household_id=hid,
            ccei_trajectory=ccei_values,
            mean_ccei=np.mean(ccei_values),
            std_ccei=np.std(ccei_values),
            slope=slope,
            trajectory_type=ttype,
        ))
    return results


# =============================================================================
# Reporting
# =============================================================================

def print_banner(text: str, char: str = "=", width: int = 70) -> None:
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_batch_results(results: list[dict], wall_time: float) -> None:
    n = len(results)
    cceis = np.array([r["ccei"] for r in results])
    mpis = np.array([r["mpi"] for r in results])
    hms = np.array([r["hm_removed"] for r in results])
    consistent = sum(1 for r in results if r["garp"])

    print_banner("SCORE DISTRIBUTIONS")
    print(f"  Data: Dunnhumby 'The Complete Journey'")
    print(f"  Households: {n}  |  GARP-consistent: {consistent} ({consistent/n*100:.1f}%)")

    print(f"\n  {'Metric':<12s} {'Mean':>8s} {'Std':>8s} {'P10':>8s}"
          f" {'P25':>8s} {'P50':>8s} {'P75':>8s} {'P90':>8s}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, arr in [("CCEI", cceis), ("MPI", mpis), ("HM removed", hms)]:
        print(f"  {label:<12s} {arr.mean():8.3f} {arr.std():8.3f}"
              f" {np.percentile(arr, 10):8.3f} {np.percentile(arr, 25):8.3f}"
              f" {np.percentile(arr, 50):8.3f} {np.percentile(arr, 75):8.3f}"
              f" {np.percentile(arr, 90):8.3f}")

    sorted_by_ccei = sorted(results, key=lambda r: r["ccei"], reverse=True)
    for label, subset in [("TOP 5 MOST RATIONAL", sorted_by_ccei[:5]),
                          ("TOP 5 LEAST RATIONAL", sorted_by_ccei[-5:])]:
        print_banner(label)
        print(f"  {'ID':<12s} {'Weeks':>5s} {'GARP':>6s} {'CCEI':>8s}"
              f" {'MPI':>8s} {'HM':>8s}")
        for r in subset:
            garp_str = "PASS" if r["garp"] else "FAIL"
            print(f"  {r['hid']:<12s} {r['T']:5d} {garp_str:>6s}"
                  f" {r['ccei']:8.3f} {r['mpi']:8.3f} {r['hm_removed']:8.3f}")


def print_temporal_results(temporal: list[TemporalResult]) -> None:
    if not temporal:
        return

    n = len(temporal)
    print_banner("TRAJECTORY CLASSIFICATION")
    print(f"  Households with 20+ weeks: {n}")

    print(f"\n  {'Type':<16s} {'N':>5s} {'%':>7s} {'Mean CCEI':>10s}"
          f" {'Std CCEI':>10s} {'Avg Slope':>10s}")
    print(f"  {'-'*16} {'-'*5} {'-'*7} {'-'*10} {'-'*10} {'-'*10}")
    for ttype in ["stable", "improving", "deteriorating", "volatile"]:
        subset = [t for t in temporal if t.trajectory_type == ttype]
        if not subset:
            continue
        pct = len(subset) / n * 100
        print(f"  {ttype:<16s} {len(subset):5d} {pct:6.1f}%"
              f" {np.mean([t.mean_ccei for t in subset]):10.3f}"
              f" {np.mean([t.std_ccei for t in subset]):10.3f}"
              f" {np.mean([t.slope for t in subset]):+10.4f}")

    # Crossovers
    print_banner("CROSSOVER EXAMPLES")
    crossovers = []
    for t in temporal:
        if len(t.ccei_trajectory) < 4:
            continue
        mid = len(t.ccei_trajectory) // 2
        fh = np.mean(t.ccei_trajectory[:mid])
        sh = np.mean(t.ccei_trajectory[mid:])
        delta = sh - fh
        if abs(delta) > 0.05:
            crossovers.append((t, fh, sh, delta))

    crossovers.sort(key=lambda x: abs(x[3]), reverse=True)
    print(f"  {'ID':<12s} {'Type':<16s} {'1st half':>10s} {'2nd half':>10s}"
          f" {'Delta':>8s}")
    print(f"  {'-'*12} {'-'*16} {'-'*10} {'-'*10} {'-'*8}")
    for t, fh, sh, delta in crossovers[:10]:
        print(f"  {t.household_id:<12s} {t.trajectory_type:<16s}"
              f" {fh:10.3f} {sh:10.3f} {delta:+8.3f}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grocery Scanner Panel — Dunnhumby Household Rationality"
    )
    parser.add_argument("--households", type=int, default=None,
                        help="Max households (default: all ~2,222)")
    parser.add_argument("--plot", action="store_true",
                        help="Save CCEI histogram (requires matplotlib)")
    args = parser.parse_args()

    print_banner("GROCERY SCANNER: DUNNHUMBY REVEALED PREFERENCE PANEL")
    print(f"  Papers: Dean & Martin (2016, AER) + Echenique et al. (2011, JPE)")
    print(f"  Data: Dunnhumby 'The Complete Journey' (2,222 households, 104 weeks)")
    print(f"  Pipeline: Rust Engine → GARP/CCEI/MPI/HM → temporal panel")
    print("=" * 70)

    # Load real data
    print_banner("[1/3] LOADING DUNNHUMBY DATA", "-", 60)
    household_logs = load_dunnhumby(max_households=args.households)
    print(f"  Loaded {len(household_logs)} households")

    # Batch score with Rust Engine
    print_banner("[2/3] BATCH SCORING (RUST ENGINE)", "-", 60)
    t0 = time.perf_counter()
    results = run_engine_batch(household_logs)
    wall_time = time.perf_counter() - t0
    print_batch_results(results, wall_time)

    # Temporal analysis
    print_banner("[3/3] TEMPORAL PANEL ANALYSIS", "-", 60)
    print(f"  Computing rolling-window CCEI (window=20, step=5)...")
    temporal = run_temporal_analysis(household_logs, window=20, step=5)
    print_temporal_results(temporal)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            cceis = [r["ccei"] for r in results]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(cceis, bins=40, alpha=0.7, color="#3498db")
            ax.set_xlabel("CCEI (Afriat Efficiency Index)")
            ax.set_ylabel("Count")
            ax.set_title(f"Dunnhumby: {len(results)} Households")
            ax.set_xlim(0, 1.05)
            plt.tight_layout()
            plt.savefig("applications/grocery_ccei_histogram.png", dpi=150)
            print(f"\n  Plot saved to applications/grocery_ccei_histogram.png")
            plt.close()
        except ImportError:
            print("  [matplotlib not installed]")

    print_banner("DONE", "=", 70)


if __name__ == "__main__":
    main()
