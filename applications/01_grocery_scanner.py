#!/usr/bin/env python3
"""Application: Grocery Scanner Panel Data — Rationality Scoring at Scale.

Replicates the revealed preference analysis from:
  Dean & Martin (2016) "Measuring Rationality with the Minimum Cost of
  Revealed Preference Violations," AER 106(11), 3297-3329.
  Echenique, Lee & Shum (2011) "The Money Pump as a Measure of Revealed
  Preference Violations," JPE 119(6), 1201-1223.

A grocery chain with loyalty-card scanner data scores each household's
weekly shopping trips for economic rationality. Prices are per-category
average prices that week; quantities are units purchased.

Data source: Uses Dunnhumby "The Complete Journey" dataset if available
(2,222 households, 2 years, 10 product categories). Falls back to
simulated data if the dataset hasn't been downloaded.

Pipeline: BehaviorLog -> GARP -> CCEI -> MPI -> Houtman-Maks -> segment.

Includes panel temporal analysis: rolling-window CCEI tracking per household,
trajectory classification (stable/improving/deteriorating/volatile), and
crossover detection.

Usage:
    python applications/01_grocery_scanner.py
    python applications/01_grocery_scanner.py --households 500 --seed 123
    python applications/01_grocery_scanner.py --simulate  # force simulation
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyrevealed import BehaviorLog, validate_consistency, compute_integrity_score
from pyrevealed import compute_confusion_metric
from pyrevealed.algorithms.mpi import compute_houtman_maks_index


# =============================================================================
# Configuration
# =============================================================================

REPO_ROOT = Path(__file__).parent.parent
DUNNHUMBY_DIR = REPO_ROOT / "dunnhumby"
TRANSACTION_FILE = DUNNHUMBY_DIR / "data" / "transaction_data.csv"

CATEGORY_NAMES = ["Soda", "Milk", "Bread", "Cheese", "Chips",
                  "Soup", "Yogurt", "Beef", "Pizza", "Lunchmeat"]
BASE_PRICES = np.array([3.50, 4.00, 3.00, 6.50, 4.50,
                         2.50, 5.00, 8.00, 7.00, 5.50])
N_GOODS = len(CATEGORY_NAMES)


# =============================================================================
# Real Data Loading (Dunnhumby)
# =============================================================================

def load_dunnhumby(max_households: int | None = None) -> list[tuple[str, BehaviorLog]]:
    """Load real Dunnhumby grocery data. Requires the dataset to be downloaded.

    Returns list of (household_id, BehaviorLog).
    """
    # Add dunnhumby dir to path for its local imports
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

    # Convert to list of (id, BehaviorLog)
    households = []
    for hh_key, hh_data in households_dict.items():
        households.append((f"HH-{hh_key}", hh_data.behavior_log))
        if max_households and len(households) >= max_households:
            break

    return households


# =============================================================================
# Simulated Data (Fallback)
# =============================================================================

def generate_weekly_prices(n_weeks: int, rng: np.random.Generator) -> np.ndarray:
    """Generate realistic weekly grocery prices with seasonal and promo effects."""
    prices = np.zeros((n_weeks, N_GOODS))
    for t in range(n_weeks):
        seasonal = 1.0 + 0.10 * np.sin(2 * np.pi * t / 52
                                         + rng.uniform(0, 2 * np.pi, N_GOODS))
        noise = rng.lognormal(0.0, 0.05, N_GOODS)
        promo = np.ones(N_GOODS)
        for g in range(N_GOODS):
            if rng.random() < 0.15:
                promo[g] = rng.uniform(0.75, 0.90)
        prices[t] = BASE_PRICES * seasonal * noise * promo
    return prices


def simulate_household(
    household_type: str, prices: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one household's weekly purchases via Cobb-Douglas + noise."""
    n_weeks = prices.shape[0]
    alpha = rng.dirichlet(np.ones(N_GOODS) * 2.0)
    budgets = rng.lognormal(np.log(80.0), 0.3, size=n_weeks)

    quantities = np.zeros((n_weeks, N_GOODS))
    if household_type == "rational":
        for t in range(n_weeks):
            quantities[t] = (alpha * budgets[t]) / prices[t]
    elif household_type == "noisy":
        sigma = rng.uniform(0.05, 0.30)
        for t in range(n_weeks):
            base = (alpha * budgets[t]) / prices[t]
            quantities[t] = base * rng.lognormal(0.0, sigma, N_GOODS)
    else:  # erratic
        for t in range(n_weeks):
            shares = rng.dirichlet(np.ones(N_GOODS))
            quantities[t] = (shares * budgets[t]) / prices[t]
    return quantities


def simulate_panel(
    n_households: int, n_weeks: int, seed: int
) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Simulate a full panel. Returns (id, type, prices, quantities)."""
    rng = np.random.default_rng(seed)
    prices = generate_weekly_prices(n_weeks, rng)

    households = []
    for i in range(n_households):
        r = rng.random()
        htype = "rational" if r < 0.40 else ("noisy" if r < 0.80 else "erratic")
        quantities = simulate_household(htype, prices, rng)
        households.append((f"HH-{i+1:04d}", htype, prices, quantities))
    return households


# =============================================================================
# Analysis
# =============================================================================

@dataclass
class HouseholdResult:
    household_id: str
    household_type: str
    n_weeks: int
    is_consistent: bool
    ccei: float
    mpi: float
    hm_fraction: float
    time_ms: float


def analyze_household(
    hid: str, log: BehaviorLog, htype: str = "unknown"
) -> HouseholdResult:
    """Run GARP -> CCEI -> MPI -> HM pipeline on one household."""
    t0 = time.perf_counter()

    garp = validate_consistency(log)

    if garp.is_consistent:
        ccei_val, mpi_val, hm_val = 1.0, 0.0, 0.0
    else:
        ccei_val = compute_integrity_score(log, tolerance=1e-4).efficiency_index
        mpi_val = compute_confusion_metric(log).mpi_value
        hm_val = compute_houtman_maks_index(log).fraction

    elapsed = (time.perf_counter() - t0) * 1000

    return HouseholdResult(
        household_id=hid, household_type=htype,
        n_weeks=log.num_records, is_consistent=garp.is_consistent,
        ccei=ccei_val, mpi=mpi_val, hm_fraction=hm_val, time_ms=elapsed,
    )


# =============================================================================
# Temporal Panel Analysis
# =============================================================================

@dataclass
class TemporalResult:
    household_id: str
    ccei_trajectory: list[float]  # CCEI per window
    window_starts: list[int]      # start index of each window
    mean_ccei: float
    std_ccei: float
    slope: float                  # linear trend
    trajectory_type: str          # stable/improving/deteriorating/volatile


def compute_rolling_ccei(
    log: BehaviorLog, window: int = 20, step: int = 5,
) -> list[tuple[int, float]]:
    """Compute CCEI over rolling windows of observations.

    Returns list of (window_start_index, ccei_value).
    """
    T = log.num_records
    if T < window:
        return [(0, compute_integrity_score(log, tolerance=1e-4).efficiency_index)]

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
        results.append((start, ccei))
    return results


def classify_trajectory(ccei_values: list[float]) -> tuple[str, float]:
    """Classify a CCEI trajectory and compute linear slope.

    Returns (trajectory_type, slope).
    """
    if len(ccei_values) < 2:
        return "stable", 0.0

    arr = np.array(ccei_values)
    std = arr.std()
    # Linear regression: slope via least squares
    x = np.arange(len(arr))
    slope = np.polyfit(x, arr, 1)[0]

    if std < 0.03:
        return "stable", slope
    elif slope > 0.005:
        return "improving", slope
    elif slope < -0.005:
        return "deteriorating", slope
    else:
        return "volatile", slope


def run_temporal_analysis(
    household_logs: list[tuple[str, BehaviorLog]],
    window: int = 20, step: int = 5,
) -> list[TemporalResult]:
    """Run rolling-window CCEI analysis for all households."""
    results = []
    for i, (hid, log) in enumerate(household_logs):
        if log.num_records < window:
            continue  # skip households with too few observations
        rolling = compute_rolling_ccei(log, window, step)
        ccei_values = [c for _, c in rolling]
        starts = [s for s, _ in rolling]
        ttype, slope = classify_trajectory(ccei_values)
        results.append(TemporalResult(
            household_id=hid,
            ccei_trajectory=ccei_values,
            window_starts=starts,
            mean_ccei=np.mean(ccei_values),
            std_ccei=np.std(ccei_values),
            slope=slope,
            trajectory_type=ttype,
        ))
        if (i + 1) % 25 == 0:
            print(f"    Temporal: {i+1}/{len(household_logs)} households...")
    return results


def print_temporal_results(temporal: list[TemporalResult]) -> None:
    """Print temporal panel analysis results."""
    if not temporal:
        print("  No households with enough observations for temporal analysis.")
        return

    n = len(temporal)
    type_counts: dict[str, int] = {}
    for t in temporal:
        type_counts[t.trajectory_type] = type_counts.get(t.trajectory_type, 0) + 1

    print_banner("TRAJECTORY CLASSIFICATION")
    print(f"  Households analyzed: {n}")
    print(f"\n  {'Type':<16s} {'N':>5s} {'%':>7s} {'Mean CCEI':>10s}"
          f" {'Std CCEI':>10s} {'Avg Slope':>10s}")
    print(f"  {'-'*16} {'-'*5} {'-'*7} {'-'*10} {'-'*10} {'-'*10}")
    for ttype in ["stable", "improving", "deteriorating", "volatile"]:
        subset = [t for t in temporal if t.trajectory_type == ttype]
        if not subset:
            continue
        count = len(subset)
        pct = count / n * 100
        avg_ccei = np.mean([t.mean_ccei for t in subset])
        avg_std = np.mean([t.std_ccei for t in subset])
        avg_slope = np.mean([t.slope for t in subset])
        print(f"  {ttype:<16s} {count:5d} {pct:6.1f}% {avg_ccei:10.3f}"
              f" {avg_std:10.3f} {avg_slope:+10.4f}")

    # Crossover examples: users whose first-half and second-half classifications differ
    print_banner("CROSSOVER EXAMPLES")
    print("  Households whose rationality changed significantly over time:")
    print(f"\n  {'ID':<12s} {'Type':<16s} {'1st half':>10s} {'2nd half':>10s}"
          f" {'Delta':>8s} {'Slope':>8s}")
    print(f"  {'-'*12} {'-'*16} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    crossovers = []
    for t in temporal:
        if len(t.ccei_trajectory) < 4:
            continue
        mid = len(t.ccei_trajectory) // 2
        first_half = np.mean(t.ccei_trajectory[:mid])
        second_half = np.mean(t.ccei_trajectory[mid:])
        delta = second_half - first_half
        if abs(delta) > 0.05:
            crossovers.append((t, first_half, second_half, delta))

    crossovers.sort(key=lambda x: abs(x[3]), reverse=True)
    for t, fh, sh, delta in crossovers[:8]:
        print(f"  {t.household_id:<12s} {t.trajectory_type:<16s}"
              f" {fh:10.3f} {sh:10.3f} {delta:+8.3f} {t.slope:+8.4f}")

    if not crossovers:
        print("  (No significant crossovers found)")


# =============================================================================
# Reporting
# =============================================================================

def print_banner(text: str, char: str = "=", width: int = 70) -> None:
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_results(results: list[HouseholdResult], wall_time: float,
                  data_source: str) -> None:
    n = len(results)
    cceis = np.array([r.ccei for r in results])
    mpis = np.array([r.mpi for r in results])
    hms = np.array([r.hm_fraction for r in results])
    consistent = sum(1 for r in results if r.is_consistent)

    print_banner("SCORE DISTRIBUTIONS")
    print(f"  Data source: {data_source}")
    print(f"  Households: {n}  |  GARP-consistent: {consistent} ({consistent/n*100:.1f}%)")
    print(f"  Wall time: {wall_time:.2f}s  |  Throughput: {n/wall_time:.0f} households/sec")

    print(f"\n  {'Metric':<12s} {'Mean':>8s} {'Std':>8s} {'P10':>8s}"
          f" {'P25':>8s} {'P50':>8s} {'P75':>8s} {'P90':>8s}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, arr in [("CCEI", cceis), ("MPI", mpis), ("HM removed", hms)]:
        print(f"  {label:<12s} {arr.mean():8.3f} {arr.std():8.3f}"
              f" {np.percentile(arr, 10):8.3f} {np.percentile(arr, 25):8.3f}"
              f" {np.percentile(arr, 50):8.3f} {np.percentile(arr, 75):8.3f}"
              f" {np.percentile(arr, 90):8.3f}")

    # Breakdown by type (only meaningful for simulated data)
    types = sorted(set(r.household_type for r in results))
    if types != ["unknown"]:
        print_banner("BREAKDOWN BY HOUSEHOLD TYPE")
        print(f"  {'Type':<12s} {'N':>5s} {'GARP%':>7s} {'CCEI':>8s}"
              f" {'MPI':>8s} {'HM':>8s}")
        print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
        for t in types:
            subset = [r for r in results if r.household_type == t]
            n_t = len(subset)
            garp_pct = sum(1 for r in subset if r.is_consistent) / n_t * 100
            print(f"  {t:<12s} {n_t:5d} {garp_pct:6.1f}%"
                  f" {np.mean([r.ccei for r in subset]):8.3f}"
                  f" {np.mean([r.mpi for r in subset]):8.3f}"
                  f" {np.mean([r.hm_fraction for r in subset]):8.3f}")

    # Top and bottom 5
    sorted_by_ccei = sorted(results, key=lambda r: r.ccei, reverse=True)
    for label, subset in [("TOP 5 MOST RATIONAL", sorted_by_ccei[:5]),
                          ("TOP 5 LEAST RATIONAL", sorted_by_ccei[-5:])]:
        print_banner(label)
        print(f"  {'ID':<12s} {'Weeks':>5s} {'GARP':>6s} {'CCEI':>8s}"
              f" {'MPI':>8s} {'HM':>8s}")
        for r in subset:
            garp_str = "PASS" if r.is_consistent else "FAIL"
            print(f"  {r.household_id:<12s} {r.n_weeks:5d} {garp_str:>6s}"
                  f" {r.ccei:8.3f} {r.mpi:8.3f} {r.hm_fraction:8.3f}")

    print_banner("INTERPRETATION")
    print("""
  Reference: Dean & Martin (2016, AER) ran GARP on 977 households' grocery
  scanner data across 38 product categories. Echenique, Lee & Shum (2011, JPE)
  computed MPI on similar data.

  Business applications:
  - Segment customers by CCEI for targeted pricing (high-CCEI = price-sensitive).
  - Flag accounts with sudden CCEI drops (potential fraud / account sharing).
  - Use MPI to quantify welfare losses from inconsistent choices.
  - Houtman-Maks identifies which specific shopping trips are "outliers."
""")


def plot_results(results: list[HouseholdResult]) -> None:
    """Optional: CCEI histogram."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib not installed -- skipping plot]")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    types = sorted(set(r.household_type for r in results))
    colors = {"rational": "#2ecc71", "noisy": "#f39c12", "erratic": "#e74c3c",
              "unknown": "#3498db"}
    for htype in types:
        cceis = [r.ccei for r in results if r.household_type == htype]
        label = htype.capitalize() if htype != "unknown" else "Dunnhumby household"
        ax.hist(cceis, bins=30, alpha=0.6, label=label,
                color=colors.get(htype, "#3498db"))

    ax.set_xlabel("CCEI (Afriat Efficiency Index)")
    ax.set_ylabel("Count")
    ax.set_title("Household Rationality Distribution -- Grocery Scanner Data")
    ax.legend()
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    out = Path("applications/grocery_ccei_histogram.png")
    plt.savefig(out, dpi=150)
    print(f"  Plot saved to {out}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grocery Scanner Panel -- Household Rationality Scoring"
    )
    parser.add_argument("--households", type=int, default=200,
                        help="Max households to analyze (default: 200)")
    parser.add_argument("--weeks", type=int, default=50,
                        help="Shopping weeks per household for simulation (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for simulation (default: 42)")
    parser.add_argument("--simulate", action="store_true",
                        help="Force simulation even if Dunnhumby data exists")
    parser.add_argument("--plot", action="store_true",
                        help="Save CCEI histogram (requires matplotlib)")
    args = parser.parse_args()

    print_banner("GROCERY SCANNER PANEL: REVEALED PREFERENCE AT SCALE")
    print(f"  Paper: Dean & Martin (2016, AER) + Echenique, Lee & Shum (2011, JPE)")
    print(f"  Pipeline: BehaviorLog -> GARP -> CCEI -> MPI -> Houtman-Maks")
    print("=" * 70)

    # Try real data first, fall back to simulation
    use_real = not args.simulate and TRANSACTION_FILE.exists()
    data_source = "Dunnhumby 'The Complete Journey'" if use_real else "Simulated"

    if use_real:
        print_banner("[1/3] LOADING DUNNHUMBY DATA", "-", 60)
        try:
            household_logs = load_dunnhumby(max_households=args.households)
            print(f"  Loaded {len(household_logs)} households from real scanner data")
        except Exception as e:
            print(f"  Failed to load Dunnhumby data: {e}")
            print("  Falling back to simulation...")
            use_real = False

    if not use_real:
        print_banner("[1/3] SIMULATING PANEL DATA", "-", 60)
        data_source = f"Simulated ({args.households} households, {args.weeks} weeks)"
        panel = simulate_panel(args.households, args.weeks, args.seed)
        types_count: dict[str, int] = {}
        for _, htype, _, _ in panel:
            types_count[htype] = types_count.get(htype, 0) + 1
        print(f"  Generated {len(panel)} households")
        for t, c in sorted(types_count.items()):
            print(f"    {t:<10s}: {c}")
        # Convert to (id, BehaviorLog) pairs with type info
        household_logs = []
        household_types: dict[str, str] = {}
        for hid, htype, prices, quantities in panel:
            log = BehaviorLog(cost_vectors=prices, action_vectors=quantities, user_id=hid)
            household_logs.append((hid, log))
            household_types[hid] = htype

    # Analyze
    print_banner("[2/3] RUNNING ANALYSIS PIPELINE", "-", 60)
    t0 = time.perf_counter()
    results = []
    for i, (hid, log) in enumerate(household_logs):
        htype = household_types.get(hid, "unknown") if not use_real else "unknown"
        result = analyze_household(hid, log, htype)
        results.append(result)
        if (i + 1) % 50 == 0 or (i + 1) == len(household_logs):
            print(f"    Processed {i+1}/{len(household_logs)} households...")
    wall_time = time.perf_counter() - t0

    # Report
    print_results(results, wall_time, data_source)

    # Temporal panel analysis
    print_banner("[3/4] TEMPORAL PANEL ANALYSIS", "-", 60)
    print(f"  Computing rolling-window CCEI (window=20, step=5)...")
    t0 = time.perf_counter()
    temporal = run_temporal_analysis(household_logs, window=20, step=5)
    temporal_time = time.perf_counter() - t0
    print(f"  Temporal analysis: {temporal_time:.2f}s for {len(temporal)} households")
    print_temporal_results(temporal)

    if args.plot:
        print_banner("[4/4] VISUALIZATION", "-", 60)
        plot_results(results)

    print_banner("DONE", "=", 70)


if __name__ == "__main__":
    main()
