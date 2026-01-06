"""Visualization and analysis of results (Phase 5).

This module generates the three showcase visualizations:

Showcase A: Rationality Histogram
    Distribution of AEI (efficiency) scores across all households.
    Business insight: Identify "erratic shoppers" with low scores.

Showcase B: Income vs. Rationality
    Scatter/box plot of income bracket vs. rationality score.
    Economic hypothesis: Lower-income households may have higher scores
    (tighter budget constraints force careful optimization).

Showcase C: Utility Recovery
    For a high-score household: recovered utility values, Lagrange multipliers,
    and demand prediction comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pyrevealed import recover_utility, UtilityRecoveryResult

from config import (
    DEMOGRAPHICS_FILE,
    HIGH_RATIONALITY_THRESHOLD,
    INCOME_ORDER,
    LOW_RATIONALITY_THRESHOLD,
    OUTPUT_DIR,
    TOP_COMMODITIES,
    COMMODITY_SHORT_NAMES,
)
from run_analysis import AnalysisResults
from session_builder import HouseholdData


def showcase_a_rationality_histogram(
    results: AnalysisResults,
    output_path: Path,
) -> None:
    """
    Showcase A: Distribution of rationality (AEI) scores across households.

    Creates a histogram showing the distribution of Afriat Efficiency Index
    scores. Most users should cluster around 0.90-0.95; a tail below 0.70
    indicates "erratic shoppers."

    Args:
        results: AnalysisResults from run_analysis
        output_path: Directory to save the plot
    """
    import matplotlib.pyplot as plt

    scores = results.get_efficiency_scores()
    stats = results.get_summary_stats()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogram
    n, bins, patches = ax.hist(
        scores,
        bins=50,
        range=(0, 1),
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )

    # Color-code bins by threshold
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i + 1]) / 2
        if bin_center < LOW_RATIONALITY_THRESHOLD:
            patch.set_facecolor("red")
        elif bin_center < HIGH_RATIONALITY_THRESHOLD:
            patch.set_facecolor("orange")
        else:
            patch.set_facecolor("green")

    # Add threshold lines
    ax.axvline(
        LOW_RATIONALITY_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Erratic (<{LOW_RATIONALITY_THRESHOLD})",
    )
    ax.axvline(
        HIGH_RATIONALITY_THRESHOLD,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Highly Rational (>{HIGH_RATIONALITY_THRESHOLD})",
    )

    # Labels and title
    ax.set_xlabel("Afriat Efficiency Index (Integrity Score)", fontsize=12)
    ax.set_ylabel("Number of Households", fontsize=12)
    ax.set_title(
        f"Household Rationality Distribution (n={len(scores):,})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add stats text box
    textstr = (
        f"Mean: {stats['mean_aei']:.3f}\n"
        f"Median: {stats['median_aei']:.3f}\n"
        f"Std Dev: {stats['std_aei']:.3f}\n"
        f"GARP Consistent: {stats['consistent_households']:,}/{stats['processed_households']:,}\n"
        f"Below 0.7: {stats['households_below_0.7']:,}\n"
        f"Perfect 1.0: {stats['households_perfect_1.0']:,}"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    output_file = output_path / "showcase_a_rationality_histogram.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file.name}")


def showcase_b_income_vs_rationality(
    results: AnalysisResults,
    output_path: Path,
) -> None:
    """
    Showcase B: Box plot of income bracket vs. rationality score.

    Tests the economic hypothesis that lower-income households may have
    higher rationality scores due to tighter budget constraints.

    Args:
        results: AnalysisResults from run_analysis
        output_path: Directory to save the plot
    """
    import matplotlib.pyplot as plt

    if not DEMOGRAPHICS_FILE.exists():
        print(f"  Skipping Showcase B: demographics file not found")
        print(f"    Expected: {DEMOGRAPHICS_FILE}")
        return

    demographics = pd.read_csv(DEMOGRAPHICS_FILE)

    # Build dataframe for plotting
    data = []
    for hh_key, result in results.household_results.items():
        demo_row = demographics[demographics["household_key"] == hh_key]
        if len(demo_row) > 0:
            income = demo_row["INCOME_DESC"].iloc[0]
            data.append(
                {
                    "household_key": hh_key,
                    "efficiency_index": result.efficiency_index,
                    "income_bracket": income,
                    "is_consistent": result.is_garp_consistent,
                }
            )

    if not data:
        print("  Skipping Showcase B: no matching demographics data")
        return

    df = pd.DataFrame(data)

    # Map income to numeric rank for ordering
    income_rank_map = {v: i for i, v in enumerate(INCOME_ORDER)}
    df["income_rank"] = df["income_bracket"].map(income_rank_map)
    df = df.dropna(subset=["income_rank"])

    if df.empty:
        print("  Skipping Showcase B: no valid income brackets found")
        return

    # Sort by income rank
    df = df.sort_values("income_rank")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Get unique income brackets in order
    brackets = df["income_bracket"].unique()
    bracket_order = sorted(brackets, key=lambda x: income_rank_map.get(x, 999))

    # Create box plot data
    box_data = [
        df[df["income_bracket"] == bracket]["efficiency_index"].values
        for bracket in bracket_order
    ]

    # Box plot
    bp = ax.boxplot(
        box_data,
        positions=range(len(bracket_order)),
        widths=0.6,
        patch_artist=True,
    )

    # Style the boxes
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    # Add mean markers
    means = [np.mean(d) for d in box_data]
    ax.scatter(
        range(len(bracket_order)),
        means,
        color="red",
        marker="D",
        s=50,
        zorder=5,
        label="Mean",
    )

    # Labels
    ax.set_xticks(range(len(bracket_order)))
    ax.set_xticklabels(bracket_order, rotation=45, ha="right")
    ax.set_xlabel("Income Bracket", fontsize=12)
    ax.set_ylabel("Afriat Efficiency Index", fontsize=12)
    ax.set_title(
        "Rationality by Income Bracket",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Add threshold line
    ax.axhline(
        LOW_RATIONALITY_THRESHOLD,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Erratic threshold ({LOW_RATIONALITY_THRESHOLD})",
    )
    ax.legend(loc="lower right")

    # Add sample sizes
    for i, bracket in enumerate(bracket_order):
        n = len(box_data[i])
        ax.text(
            i,
            -0.02,
            f"n={n}",
            ha="center",
            va="top",
            fontsize=8,
            transform=ax.get_xaxis_transform(),
        )

    plt.tight_layout()
    output_file = output_path / "showcase_b_income_vs_rationality.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file.name}")


def showcase_c_utility_recovery(
    results: AnalysisResults,
    households: Dict[int, HouseholdData],
    output_path: Path,
) -> None:
    """
    Showcase C: Utility recovery for a high-score household.

    For a perfectly consistent household:
    1. Recover utility values using linear programming
    2. Plot recovered utilities over time
    3. Plot Lagrange multipliers (marginal utility of money)
    4. Compare actual vs predicted demand

    Args:
        results: AnalysisResults from run_analysis
        households: Dict of HouseholdData
        output_path: Directory to save plots
    """
    import matplotlib.pyplot as plt

    # Find a perfectly consistent household with many observations
    consistent_results = [
        r
        for r in results.household_results.values()
        if r.is_garp_consistent and r.efficiency_index == 1.0
    ]

    if not consistent_results:
        # Fall back to highest AEI score
        consistent_results = [
            r
            for r in results.household_results.values()
            if r.is_garp_consistent and r.efficiency_index >= 0.99
        ]

    if not consistent_results:
        print("  Skipping Showcase C: no highly consistent households found")
        return

    # Pick one with most observations
    best = max(consistent_results, key=lambda r: r.num_observations)
    hh_key = best.household_key

    if hh_key not in households:
        print(f"  Skipping Showcase C: household {hh_key} not found")
        return

    hh_data = households[hh_key]
    log = hh_data.behavior_log

    print(
        f"  Selected household {hh_key} "
        f"(T={best.num_observations}, AEI={best.efficiency_index:.4f})"
    )

    # Recover utility
    utility_result: UtilityRecoveryResult = recover_utility(log)

    if not utility_result.success:
        print(f"  Utility recovery failed: {utility_result.lp_status}")
        return

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Utility values over time
    ax = axes[0, 0]
    ax.bar(
        range(len(utility_result.utility_values)),
        utility_result.utility_values,
        color="steelblue",
        alpha=0.7,
    )
    ax.set_xlabel("Observation (Shopping Week)")
    ax.set_ylabel("Utility Value")
    ax.set_title(f"Recovered Utility Values (Household {hh_key})")
    ax.grid(True, alpha=0.3)

    # Plot 2: Lagrange multipliers (marginal utility of money)
    ax = axes[0, 1]
    if utility_result.lagrange_multipliers is not None:
        ax.bar(
            range(len(utility_result.lagrange_multipliers)),
            utility_result.lagrange_multipliers,
            color="orange",
            alpha=0.7,
        )
        ax.set_xlabel("Observation (Shopping Week)")
        ax.set_ylabel("Marginal Utility of Money (λ)")
        ax.set_title("Recovered Lagrange Multipliers")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No multipliers available", ha="center", va="center")
        ax.set_title("Lagrange Multipliers")

    # Plot 3: Average quantities by commodity
    ax = axes[1, 0]
    avg_quantities = log.action_vectors.mean(axis=0)
    short_names = [COMMODITY_SHORT_NAMES.get(c, c[:8]) for c in TOP_COMMODITIES]
    bars = ax.bar(range(len(avg_quantities)), avg_quantities, color="green", alpha=0.7)
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_xlabel("Commodity")
    ax.set_ylabel("Average Quantity")
    ax.set_title("Average Purchase Quantities by Commodity")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Expenditure over time
    ax = axes[1, 1]
    expenditures = log.total_spend
    ax.plot(range(len(expenditures)), expenditures, "b-o", markersize=3, alpha=0.7)
    ax.axhline(np.mean(expenditures), color="red", linestyle="--", label="Mean")
    ax.set_xlabel("Observation (Shopping Week)")
    ax.set_ylabel("Total Expenditure ($)")
    ax.set_title("Expenditure Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Utility Analysis: Household {hh_key} (AEI={best.efficiency_index:.3f})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_file = output_path / "showcase_c_utility_recovery.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved: {output_file.name}")


def generate_all_visualizations(
    results: AnalysisResults,
    households: Dict[int, HouseholdData],
    output_path: Optional[Path] = None,
) -> None:
    """
    Generate all showcase visualizations.

    Args:
        results: AnalysisResults from run_analysis
        households: Dict of HouseholdData
        output_path: Directory to save plots (default: OUTPUT_DIR)
    """
    if output_path is None:
        output_path = OUTPUT_DIR

    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating visualizations...")

    try:
        showcase_a_rationality_histogram(results, output_path)
    except Exception as e:
        print(f"  Error generating Showcase A: {e}")

    try:
        showcase_b_income_vs_rationality(results, output_path)
    except Exception as e:
        print(f"  Error generating Showcase B: {e}")

    try:
        showcase_c_utility_recovery(results, households, output_path)
    except Exception as e:
        print(f"  Error generating Showcase C: {e}")

    print(f"\nAll visualizations saved to: {output_path}")


def save_results_csv(
    results: AnalysisResults,
    output_path: Optional[Path] = None,
) -> None:
    """
    Save analysis results to CSV for further analysis.

    Args:
        results: AnalysisResults from run_analysis
        output_path: Directory to save CSV (default: OUTPUT_DIR)
    """
    if output_path is None:
        output_path = OUTPUT_DIR

    output_path.mkdir(parents=True, exist_ok=True)

    df = results.to_dataframe()
    csv_file = output_path / "household_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"  Results saved to: {csv_file}")
