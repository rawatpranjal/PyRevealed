"""
Comprehensive analysis of Prest example datasets using PyRevealed.

This script loads the budgetary dataset from the prest project and runs
all available revealed preference analyses: GARP, AEI, MPI, Houtman-Maks,
and utility recovery.

Usage:
    python scripts/analyze_prest_data.py
"""

import io
import ssl
import urllib.request

import numpy as np
import pandas as pd

from pyrevealed import (
    ConsumerSession,
    check_garp,
    compute_aei,
    compute_mpi,
    recover_utility,
)
from pyrevealed.algorithms.mpi import compute_houtman_maks_index


def load_prest_datasets() -> dict[str, pd.DataFrame]:
    """Load all prest example datasets from GitHub."""
    BASE_URL = "https://raw.githubusercontent.com/prestsoftware/prest/master/docs/src/_static/examples/"

    DATASETS = [
        "budgetary.csv",
        "estimation-models-defaults.csv",
        "estimation-models-no-defaults.csv",
        "general-defaults-128.csv",
        "general-defaults.csv",
        "general-hybrid.csv",
        "general-merging.csv",
        "general-no-defaults-128.csv",
        "general-no-defaults.csv",
        "general-stochastic-consistency.csv",
        "integrity.csv",
    ]

    # Create SSL context that doesn't verify certificates (for macOS)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    datasets = {}
    for name in DATASETS:
        key = name.replace(".csv", "").replace("-", "_")
        url = BASE_URL + name
        with urllib.request.urlopen(url, context=ssl_context) as response:
            csv_data = response.read().decode("utf-8")
        datasets[key] = pd.read_csv(io.StringIO(csv_data))
        print(f"Loaded {name}: {datasets[key].shape}")

    return datasets


def analyze_budgetary_data(budgetary: pd.DataFrame) -> pd.DataFrame:
    """
    Run comprehensive revealed preference analysis on budgetary data.

    Analyses performed:
    - GARP: Check for preference violations
    - AEI: Afriat Efficiency Index (0-1 consistency score)
    - MPI: Money Pump Index (exploitability measure)
    - Houtman-Maks: Minimum observations to remove for consistency
    """
    # Dynamically detect price and demand columns
    price_cols = [c for c in budgetary.columns if c.startswith("Price")]
    demand_cols = [c for c in budgetary.columns if c.startswith("Demand")]
    print(f"Detected columns: {price_cols}, {demand_cols}")

    results = []
    sessions = {}

    for subject in budgetary["Subject"].unique():
        subject_data = budgetary[budgetary["Subject"] == subject]
        prices = subject_data[price_cols].values
        quantities = subject_data[demand_cols].values

        # Filter out zero-price columns (some subjects have fewer goods)
        valid_cols = (prices > 0).any(axis=0)
        prices = prices[:, valid_cols]
        quantities = quantities[:, valid_cols]

        session = ConsumerSession(prices=prices, quantities=quantities)
        sessions[subject] = session

        # Run all analyses
        garp_result = check_garp(session)
        aei_result = compute_aei(session)
        mpi_result = compute_mpi(session)
        hm_index, removed_obs = compute_houtman_maks_index(session)

        results.append(
            {
                "Subject": subject,
                "Observations": session.num_observations,
                "Goods": session.num_goods,
                "GARP_Consistent": garp_result.is_consistent,
                "Violations": garp_result.num_violations,
                "AEI": aei_result.efficiency_index,
                "MPI": mpi_result.mpi_value,
                "HM_Index": hm_index,
                "Removed_Obs": len(removed_obs),
            }
        )

    return pd.DataFrame(results), sessions


def recover_utilities(
    budgetary: pd.DataFrame, sessions: dict[str, ConsumerSession]
) -> None:
    """Recover utility functions for GARP-consistent subjects."""
    print("\n" + "=" * 60)
    print("UTILITY RECOVERY")
    print("=" * 60)

    for subject in budgetary["Subject"].unique():
        session = sessions[subject]
        garp_result = check_garp(session)

        if garp_result.is_consistent:
            utility_result = recover_utility(session)
            if utility_result.success:
                print(f"\n{subject}: Utility recovered successfully")
                print(f"  Utility values: {utility_result.utility_values}")
                print(
                    f"  Marginal utility of money: {utility_result.lagrange_multipliers}"
                )
            else:
                print(f"\n{subject}: Recovery failed - {utility_result.lp_status}")
        else:
            print(
                f"\n{subject}: GARP violated ({garp_result.num_violations} violations) - cannot recover utility"
            )


def visualize_results(
    results_df: pd.DataFrame, sessions: dict[str, ConsumerSession]
) -> None:
    """Generate visualizations for the analysis results."""
    import matplotlib.pyplot as plt
    from pyrevealed.viz import plot_aei_distribution, plot_budget_sets
    from pyrevealed.viz.plots import plot_preference_heatmap

    print("\nGenerating visualizations...")

    # 1. AEI Distribution across subjects
    print("  - AEI distribution plot")
    fig, ax = plot_aei_distribution(results_df["AEI"].tolist(), bins=10)
    plt.savefig("aei_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Preference heatmap for worst (lowest AEI) subject
    worst = results_df.loc[results_df["AEI"].idxmin(), "Subject"]
    print(f"  - Preference heatmap for {worst}")
    fig, ax = plot_preference_heatmap(sessions[worst], matrix_type="transitive")
    plt.savefig(f"{worst}_preference_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Budget sets for 2-good subjects
    for subject, session in sessions.items():
        if session.num_goods == 2:
            print(f"  - Budget sets for {subject}")
            fig, ax = plot_budget_sets(session)
            plt.savefig(f"{subject}_budget_sets.png", dpi=150, bbox_inches="tight")
            plt.close()

    print("Visualizations saved to current directory.")


# =============================================================================
# MENU CHOICE ANALYSIS (WARP/SARP)
# =============================================================================


def _detect_cycle(edges: list[tuple[str, str]]) -> bool:
    """Detect cycle in directed graph using DFS."""
    from collections import defaultdict

    graph: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for a, b in edges:
        graph[a].add(b)
        nodes.add(a)
        nodes.add(b)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}

    def dfs(node: str) -> bool:
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True  # Back edge = cycle
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    for node in nodes:
        if color[node] == WHITE:
            if dfs(node):
                return True
    return False


def check_warp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check WARP (Weak Axiom of Revealed Preference) for menu choice data.

    WARP: If x is chosen when y is available, then y should never
    be chosen when x is available.

    Returns DataFrame with violations per subject.
    """
    results = []
    for subject, data in df.groupby("subject"):
        violations = []
        choices: list[tuple[frozenset[str], str]] = []

        for _, row in data.iterrows():
            menu = frozenset(x.strip() for x in str(row["menu"]).split(","))
            choice = str(row["choice"]).strip()

            # Check against all previous choices
            for prev_menu, prev_choice in choices:
                # WARP violation: x chosen from {x,y}, y chosen from {y,x}
                if prev_choice in menu and choice in prev_menu and choice != prev_choice:
                    violations.append((prev_menu, prev_choice, menu, choice))

            choices.append((menu, choice))

        results.append(
            {
                "subject": subject,
                "total_choices": len(choices),
                "warp_violations": len(violations),
                "warp_consistent": len(violations) == 0,
            }
        )

    return pd.DataFrame(results)


def check_sarp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check SARP (Strong Axiom of Revealed Preference) for menu choice data.

    SARP: No cycles in the revealed preference relation.
    Uses DFS cycle detection on the preference graph.

    Returns DataFrame with SARP consistency per subject.
    """
    results = []
    for subject, data in df.groupby("subject"):
        # Build preference graph: x -> y means x revealed preferred to y
        preferences: list[tuple[str, str]] = []

        for _, row in data.iterrows():
            menu_items = [x.strip() for x in str(row["menu"]).split(",")]
            choice = str(row["choice"]).strip()
            for item in menu_items:
                if item != choice:
                    preferences.append((choice, item))

        # Check for cycles using DFS
        has_cycle = _detect_cycle(preferences) if preferences else False

        results.append(
            {
                "subject": subject,
                "total_preferences": len(preferences),
                "sarp_consistent": not has_cycle,
            }
        )

    return pd.DataFrame(results)


def analyze_default_bias(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Analyze tendency to stick with the default option.

    Only for datasets with 'default' column.
    Returns DataFrame with default bias metrics per subject.
    """
    if "default" not in df.columns:
        return None

    results = []
    for subject, data in df.groupby("subject"):
        total = len(data)
        chose_default = (data["choice"] == data["default"]).sum()

        results.append(
            {
                "subject": subject,
                "total_choices": total,
                "chose_default": int(chose_default),
                "default_rate": chose_default / total if total > 0 else 0,
            }
        )

    return pd.DataFrame(results)


def analyze_menu_choice_data(datasets: dict[str, pd.DataFrame]) -> None:
    """Run WARP/SARP analysis on menu choice datasets."""
    print("\n" + "=" * 60)
    print("MENU CHOICE ANALYSIS (WARP/SARP)")
    print("=" * 60)

    menu_datasets = [
        "integrity",
        "general_defaults",
        "general_no_defaults",
    ]

    for name in menu_datasets:
        if name not in datasets:
            continue

        df = datasets[name]
        print(f"\n{name.upper()}")
        print("-" * 40)
        print(f"Total rows: {len(df)}, Subjects: {df['subject'].nunique()}")

        # WARP analysis
        warp_results = check_warp(df)
        warp_consistent = warp_results["warp_consistent"].sum()
        total_subjects = len(warp_results)
        print(f"\nWARP: {warp_consistent}/{total_subjects} subjects consistent")
        if len(warp_results) <= 10:
            print(warp_results.to_string(index=False))
        else:
            print(f"  Total violations: {warp_results['warp_violations'].sum()}")
            print(f"  Avg violations/subject: {warp_results['warp_violations'].mean():.2f}")

        # SARP analysis
        sarp_results = check_sarp(df)
        sarp_consistent = sarp_results["sarp_consistent"].sum()
        print(f"\nSARP: {sarp_consistent}/{total_subjects} subjects consistent")
        if len(sarp_results) <= 10:
            print(sarp_results.to_string(index=False))

        # Default bias (if applicable)
        bias_results = analyze_default_bias(df)
        if bias_results is not None:
            avg_default_rate = bias_results["default_rate"].mean()
            print(f"\nDefault Bias: {avg_default_rate:.1%} average default choice rate")
            if len(bias_results) <= 10:
                print(bias_results.to_string(index=False))


def print_summary_stats(results_df: pd.DataFrame) -> None:
    """Print summary statistics for the analysis results."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    aei_scores = results_df["AEI"].values

    print(f"\nAEI Statistics:")
    print(f"  Mean:   {np.mean(aei_scores):.4f}")
    print(f"  Median: {np.median(aei_scores):.4f}")
    print(f"  Std:    {np.std(aei_scores):.4f}")
    print(f"  Min:    {np.min(aei_scores):.4f}")
    print(f"  Max:    {np.max(aei_scores):.4f}")

    print(f"\nConsistency Summary:")
    print(
        f"  GARP-consistent subjects: {results_df['GARP_Consistent'].sum()}/{len(results_df)}"
    )
    print(f"  Subjects with AEI >= 0.95: {(results_df['AEI'] >= 0.95).sum()}")
    print(f"  Subjects with AEI < 0.85:  {(results_df['AEI'] < 0.85).sum()}")

    print(f"\nMPI Statistics:")
    print(f"  Mean MPI: {results_df['MPI'].mean():.4f}")
    print(f"  Max MPI:  {results_df['MPI'].max():.4f}")


def main():
    print("=" * 60)
    print("PREST DATASET ANALYSIS WITH PYREVEALED")
    print("=" * 60)

    # Load datasets
    print("\nLoading datasets...")
    datasets = load_prest_datasets()

    # Analyze budgetary data
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS: BUDGETARY DATA")
    print("=" * 60)

    results_df, sessions = analyze_budgetary_data(datasets["budgetary"])

    print("\nResults:")
    print(results_df.to_string(index=False))

    # Print summary statistics
    print_summary_stats(results_df)

    # Recover utilities for consistent subjects
    recover_utilities(datasets["budgetary"], sessions)

    # Generate visualizations
    print("\n" + "=" * 60)
    print("VISUALIZATIONS")
    print("=" * 60)
    visualize_results(results_df, sessions)

    # Analyze menu choice datasets (WARP/SARP)
    analyze_menu_choice_data(datasets)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return results_df, sessions, datasets


if __name__ == "__main__":
    results_df, sessions, datasets = main()
