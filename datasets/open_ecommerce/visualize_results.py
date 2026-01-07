"""Visualization generation for Open E-Commerce 1.0 dataset (Phase 5).

This module creates showcase visualizations demonstrating PyRevealed's
analysis capabilities on real consumer data.

Visualizations:
- Showcase A: AEI distribution histogram
- Showcase B: Spend vs. rationality scatter plot
- Showcase C: Category preference analysis
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import CACHE_DIR
from run_analysis import AnalysisResults

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def ensure_output_dir(output_dir: Path) -> Path:
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_aei_distribution(
    results: AnalysisResults,
    output_path: Path,
    high_threshold: float = 0.95,
    low_threshold: float = 0.70,
) -> bool:
    """
    Showcase A: AEI distribution histogram.

    Shows the distribution of Afriat Efficiency Index scores across all users,
    color-coded by rationality tier.

    Args:
        results: AnalysisResults with AEI scores
        output_path: Path to save the figure
        high_threshold: Threshold for high rationality tier
        low_threshold: Threshold for low rationality tier

    Returns:
        True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  Skipping: matplotlib not installed")
        return False

    scores = results.aei_scores
    if not scores:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram with color-coded bins
    bins = np.linspace(0, 1, 41)  # 40 bins
    n, bins_edges, patches = ax.hist(scores, bins=bins, edgecolor="white", alpha=0.8)

    # Color code the bars
    for i, (patch, left_edge) in enumerate(zip(patches, bins_edges[:-1])):
        if left_edge >= high_threshold:
            patch.set_facecolor("#2ecc71")  # Green - high rationality
        elif left_edge >= low_threshold:
            patch.set_facecolor("#f39c12")  # Orange - medium rationality
        else:
            patch.set_facecolor("#e74c3c")  # Red - low rationality

    # Add vertical lines for thresholds
    ax.axvline(x=high_threshold, color="#27ae60", linestyle="--", linewidth=2, label=f"High (≥{high_threshold})")
    ax.axvline(x=low_threshold, color="#d35400", linestyle="--", linewidth=2, label=f"Low (<{low_threshold})")

    # Labels and title
    ax.set_xlabel("Afriat Efficiency Index (AEI)", fontsize=12)
    ax.set_ylabel("Number of Users", fontsize=12)
    ax.set_title("Distribution of Consumer Rationality Scores\nOpen E-Commerce 1.0 Dataset", fontsize=14)

    # Legend
    legend_elements = [
        mpatches.Patch(color="#2ecc71", label=f"High (≥{high_threshold})"),
        mpatches.Patch(color="#f39c12", label=f"Medium ({low_threshold}-{high_threshold})"),
        mpatches.Patch(color="#e74c3c", label=f"Low (<{low_threshold})"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # Add statistics text box
    tiers = results.count_by_tier(high_threshold, low_threshold)
    stats_text = (
        f"N = {results.n_users:,}\n"
        f"Mean = {results.aei_mean:.3f}\n"
        f"Median = {results.aei_median:.3f}\n"
        f"High: {tiers['high']:,} ({100*tiers['high']/results.n_users:.1f}%)\n"
        f"Medium: {tiers['medium']:,} ({100*tiers['medium']/results.n_users:.1f}%)\n"
        f"Low: {tiers['low']:,} ({100*tiers['low']/results.n_users:.1f}%)"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path.name}")
    return True


def plot_spend_vs_rationality(
    results: AnalysisResults,
    output_path: Path,
) -> bool:
    """
    Showcase B: Spend vs. rationality scatter plot.

    Shows the relationship between total spending and rationality score.

    Args:
        results: AnalysisResults with user data
        output_path: Path to save the figure

    Returns:
        True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  Skipping: matplotlib not installed")
        return False

    if not results.user_results:
        return False

    # Extract data
    spends = [r.total_spend for r in results.user_results]
    aeis = [r.aei for r in results.user_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with color based on AEI
    scatter = ax.scatter(spends, aeis, c=aeis, cmap="RdYlGn", alpha=0.6, s=20, vmin=0.5, vmax=1.0)
    plt.colorbar(scatter, label="AEI Score")

    # Add trend line
    z = np.polyfit(spends, aeis, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(spends), max(spends), 100)
    ax.plot(x_line, p(x_line), "b--", alpha=0.8, linewidth=2, label=f"Trend (slope: {z[0]:.2e})")

    # Calculate correlation
    correlation = np.corrcoef(spends, aeis)[0, 1]

    # Labels and title
    ax.set_xlabel("Total Spend ($)", fontsize=12)
    ax.set_ylabel("Afriat Efficiency Index (AEI)", fontsize=12)
    ax.set_title("Consumer Spending vs. Rationality Score\nOpen E-Commerce 1.0 Dataset", fontsize=14)
    ax.set_ylim(0, 1.05)

    # Add correlation text
    ax.text(0.98, 0.02, f"Correlation: {correlation:.3f}", transform=ax.transAxes,
            fontsize=11, horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path.name}")
    return True


def plot_observations_vs_rationality(
    results: AnalysisResults,
    output_path: Path,
) -> bool:
    """
    Showcase C: Observations vs. rationality box plot.

    Shows how the number of purchase periods affects rationality scores.

    Args:
        results: AnalysisResults with user data
        output_path: Path to save the figure

    Returns:
        True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  Skipping: matplotlib not installed")
        return False

    if not results.user_results:
        return False

    # Create bins based on observation counts
    df = results.to_dataframe()

    # Create observation bins
    bins = [0, 10, 20, 30, 40, 50, 100]
    labels = ["1-10", "11-20", "21-30", "31-40", "41-50", "50+"]
    df["obs_bin"] = pd.cut(df["num_observations"], bins=bins, labels=labels, right=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot
    box_data = [df[df["obs_bin"] == label]["aei"].dropna().values for label in labels]
    box_data = [d for d in box_data if len(d) > 0]  # Remove empty bins
    valid_labels = [label for label, d in zip(labels, [df[df["obs_bin"] == label]["aei"].dropna().values for label in labels]) if len(d) > 0]

    bp = ax.boxplot(box_data, labels=valid_labels, patch_artist=True)

    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(box_data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Labels and title
    ax.set_xlabel("Number of Purchase Periods (Months)", fontsize=12)
    ax.set_ylabel("Afriat Efficiency Index (AEI)", fontsize=12)
    ax.set_title("Rationality Score by Shopping Frequency\nOpen E-Commerce 1.0 Dataset", fontsize=14)
    ax.set_ylim(0, 1.05)

    # Add sample size annotations
    for i, (label, data) in enumerate(zip(valid_labels, box_data)):
        ax.annotate(f"n={len(data)}", xy=(i + 1, 0.02), ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path.name}")
    return True


def plot_consistency_by_spend_quartile(
    results: AnalysisResults,
    output_path: Path,
) -> bool:
    """
    Showcase D: GARP consistency rate by spending quartile.

    Args:
        results: AnalysisResults with user data
        output_path: Path to save the figure

    Returns:
        True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  Skipping: matplotlib not installed")
        return False

    if not results.user_results:
        return False

    df = results.to_dataframe()

    # Create spend quartiles
    df["spend_quartile"] = pd.qcut(df["total_spend"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])

    # Calculate consistency rate per quartile
    consistency_by_quartile = df.groupby("spend_quartile", observed=True)["is_consistent"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(consistency_by_quartile.index, consistency_by_quartile.values,
                  color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"], alpha=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, consistency_by_quartile.values):
        ax.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, val),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Spending Quartile", fontsize=12)
    ax.set_ylabel("GARP Consistency Rate (%)", fontsize=12)
    ax.set_title("Consistency Rate by Spending Level\nOpen E-Commerce 1.0 Dataset", fontsize=14)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path.name}")
    return True


def generate_all_visualizations(
    results: AnalysisResults,
    output_dir: Optional[Path] = None,
) -> int:
    """
    Generate all showcase visualizations.

    Args:
        results: AnalysisResults with all user data
        output_dir: Directory to save figures (default: datasets/open_ecommerce/output/)

    Returns:
        Number of visualizations successfully generated
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  Warning: matplotlib not installed, skipping visualizations")
        print("  Install with: pip install matplotlib")
        return 0

    if output_dir is None:
        output_dir = Path(__file__).parent / "output"

    output_dir = ensure_output_dir(output_dir)

    success_count = 0

    # Showcase A: AEI Distribution
    if plot_aei_distribution(results, output_dir / "aei_distribution.png"):
        success_count += 1

    # Showcase B: Spend vs. Rationality
    if plot_spend_vs_rationality(results, output_dir / "spend_vs_rationality.png"):
        success_count += 1

    # Showcase C: Observations vs. Rationality
    if plot_observations_vs_rationality(results, output_dir / "observations_vs_rationality.png"):
        success_count += 1

    # Showcase D: Consistency by Quartile
    if plot_consistency_by_spend_quartile(results, output_dir / "consistency_by_quartile.png"):
        success_count += 1

    return success_count


if __name__ == "__main__":
    from session_builder import load_sessions
    from run_analysis import run_analysis

    print("Generating visualizations for Open E-Commerce 1.0...")

    # Load and analyze users
    users = load_sessions(use_cache=True)
    results = run_analysis(users, max_users=500, progress_interval=100)

    # Generate visualizations
    n_viz = generate_all_visualizations(results)
    print(f"\nGenerated {n_viz} visualizations")
