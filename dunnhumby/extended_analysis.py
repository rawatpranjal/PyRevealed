#!/usr/bin/env python3
"""Extended analysis for Dunnhumby case study.

Generates additional insights:
1. Income correlation analysis
2. Time trends analysis
3. Spending patterns analysis
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import DEMOGRAPHICS_FILE, INCOME_ORDER, OUTPUT_DIR


def load_results() -> pd.DataFrame:
    """Load household results from CSV."""
    results_file = OUTPUT_DIR / "household_results.csv"
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}\nRun run_all.py first.")
    return pd.read_csv(results_file)


def load_demographics() -> pd.DataFrame:
    """Load household demographics."""
    if not DEMOGRAPHICS_FILE.exists():
        raise FileNotFoundError(f"Demographics not found: {DEMOGRAPHICS_FILE}")
    return pd.read_csv(DEMOGRAPHICS_FILE)


def analyze_income_correlation(results: pd.DataFrame, demographics: pd.DataFrame) -> dict:
    """
    Analyze correlation between income and rationality.

    Hypothesis: Lower-income households may have higher integrity scores
    due to tighter budget constraints forcing careful optimization.
    """
    import matplotlib.pyplot as plt

    # Merge results with demographics
    merged = results.merge(demographics[['household_key', 'INCOME_DESC']], on='household_key', how='inner')

    # Map income to numeric rank
    income_rank_map = {v: i for i, v in enumerate(INCOME_ORDER)}
    merged['income_rank'] = merged['INCOME_DESC'].map(income_rank_map)
    merged = merged.dropna(subset=['income_rank'])

    # Spearman correlation (rank-based, handles ordinal income)
    corr, p_value = stats.spearmanr(merged['income_rank'], merged['efficiency_index'])

    # Group statistics
    income_stats = merged.groupby('INCOME_DESC').agg({
        'efficiency_index': ['mean', 'std', 'count'],
        'income_rank': 'first'
    }).round(4)
    income_stats.columns = ['mean_aei', 'std_aei', 'count', 'rank']
    income_stats = income_stats.sort_values('rank')

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    ax = axes[0]
    brackets = [b for b in INCOME_ORDER if b in merged['INCOME_DESC'].values]
    box_data = [merged[merged['INCOME_DESC'] == b]['efficiency_index'].values for b in brackets]
    bp = ax.boxplot(box_data, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xticklabels(brackets, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Income Bracket')
    ax.set_ylabel('Integrity Score (AEI)')
    ax.set_title(f'Rationality by Income\n(Spearman r={corr:.3f}, p={p_value:.4f})')
    ax.grid(True, alpha=0.3, axis='y')

    # Scatter with trend line
    ax = axes[1]
    ax.scatter(merged['income_rank'], merged['efficiency_index'], alpha=0.3, s=10)
    z = np.polyfit(merged['income_rank'], merged['efficiency_index'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged['income_rank'].min(), merged['income_rank'].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.4f})')
    ax.set_xlabel('Income Rank (0=lowest, 10=highest)')
    ax.set_ylabel('Integrity Score (AEI)')
    ax.set_title('Income vs Rationality Trend')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analysis_income_correlation.png', dpi=150)
    plt.close()

    return {
        'spearman_correlation': float(corr),
        'p_value': float(p_value),
        'n_samples': len(merged),
        'significant': p_value < 0.05,
        'interpretation': 'Higher income → slightly higher rationality' if corr > 0 else 'Higher income → slightly lower rationality',
        'income_stats': income_stats.to_dict(),
    }


def analyze_spending_patterns(results: pd.DataFrame) -> dict:
    """
    Analyze correlation between total spending and rationality.

    Question: Do big spenders behave differently than small spenders?
    """
    import matplotlib.pyplot as plt

    # Pearson correlation
    corr, p_value = stats.pearsonr(results['total_spend'], results['efficiency_index'])

    # Spending quartiles
    results['spend_quartile'] = pd.qcut(results['total_spend'], 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    quartile_stats = results.groupby('spend_quartile')['efficiency_index'].agg(['mean', 'std', 'count'])

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(results['total_spend'], results['efficiency_index'], alpha=0.4, s=15, c='steelblue')

    # Add trend line
    z = np.polyfit(results['total_spend'], results['efficiency_index'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(results['total_spend'].min(), results['total_spend'].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (r={corr:.3f})')

    ax.set_xlabel('Total Spending ($)')
    ax.set_ylabel('Integrity Score (AEI)')
    ax.set_title(f'Spending vs Rationality\n(Pearson r={corr:.3f}, p={p_value:.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot by quartile
    ax = axes[1]
    quartiles = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    box_data = [results[results['spend_quartile'] == q]['efficiency_index'].values for q in quartiles]
    bp = ax.boxplot(box_data, patch_artist=True)
    colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(quartiles)
    ax.set_xlabel('Spending Quartile')
    ax.set_ylabel('Integrity Score (AEI)')
    ax.set_title('Rationality by Spending Level')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analysis_spending_patterns.png', dpi=150)
    plt.close()

    return {
        'pearson_correlation': float(corr),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'interpretation': 'Higher spending → higher rationality' if corr > 0 else 'Higher spending → lower rationality',
        'quartile_stats': quartile_stats.to_dict(),
    }


def analyze_observation_effect(results: pd.DataFrame) -> dict:
    """
    Analyze if more observations (longer history) correlates with rationality.

    This proxies for "time trends" - households with more observations
    have been shopping longer, so we can see if experience matters.
    """
    import matplotlib.pyplot as plt

    # Correlation between number of observations and AEI
    corr, p_value = stats.pearsonr(results['num_observations'], results['efficiency_index'])

    # Group by observation count ranges
    results['obs_group'] = pd.cut(results['num_observations'],
                                   bins=[0, 20, 40, 60, 80, 100],
                                   labels=['10-20', '21-40', '41-60', '61-80', '81-100'])
    obs_stats = results.groupby('obs_group')['efficiency_index'].agg(['mean', 'std', 'count'])

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    ax = axes[0]
    ax.scatter(results['num_observations'], results['efficiency_index'], alpha=0.4, s=15, c='green')
    z = np.polyfit(results['num_observations'], results['efficiency_index'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(results['num_observations'].min(), results['num_observations'].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (r={corr:.3f})')
    ax.set_xlabel('Number of Observations (Shopping Weeks)')
    ax.set_ylabel('Integrity Score (AEI)')
    ax.set_title(f'Shopping History Length vs Rationality\n(Pearson r={corr:.3f}, p={p_value:.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot by observation group
    ax = axes[1]
    groups = ['10-20', '21-40', '41-60', '61-80', '81-100']
    box_data = [results[results['obs_group'] == g]['efficiency_index'].dropna().values for g in groups]
    box_data = [d for d in box_data if len(d) > 0]
    valid_groups = [g for g, d in zip(groups, [results[results['obs_group'] == g]['efficiency_index'].dropna().values for g in groups]) if len(d) > 0]

    bp = ax.boxplot(box_data, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    ax.set_xticklabels(valid_groups)
    ax.set_xlabel('Number of Shopping Weeks')
    ax.set_ylabel('Integrity Score (AEI)')
    ax.set_title('Rationality by Shopping History Length')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analysis_time_trends.png', dpi=150)
    plt.close()

    return {
        'pearson_correlation': float(corr),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'interpretation': 'More shopping history → higher rationality' if corr > 0 else 'More shopping history → lower rationality',
        'observation_stats': obs_stats.to_dict(),
    }


def generate_summary_statistics(results: pd.DataFrame) -> dict:
    """Generate comprehensive summary statistics."""
    return {
        'total_households': len(results),
        'consistent_count': int(results['is_garp_consistent'].sum()),
        'consistent_pct': float(results['is_garp_consistent'].mean() * 100),
        'mean_aei': float(results['efficiency_index'].mean()),
        'median_aei': float(results['efficiency_index'].median()),
        'std_aei': float(results['efficiency_index'].std()),
        'min_aei': float(results['efficiency_index'].min()),
        'max_aei': float(results['efficiency_index'].max()),
        'below_0.7': int((results['efficiency_index'] < 0.7).sum()),
        'below_0.7_pct': float((results['efficiency_index'] < 0.7).mean() * 100),
        'perfect_1.0': int((results['efficiency_index'] == 1.0).sum()),
        'perfect_1.0_pct': float((results['efficiency_index'] == 1.0).mean() * 100),
        'mean_observations': float(results['num_observations'].mean()),
        'mean_spend': float(results['total_spend'].mean()),
        'total_spend': float(results['total_spend'].sum()),
    }


def run_extended_analysis() -> dict:
    """Run all extended analyses and return results."""
    print("=" * 60)
    print(" EXTENDED DUNNHUMBY ANALYSIS")
    print("=" * 60)

    results = load_results()
    print(f"\nLoaded {len(results)} household results")

    all_results = {}

    # Summary stats
    print("\n[1/4] Generating summary statistics...")
    all_results['summary'] = generate_summary_statistics(results)

    # Income correlation
    print("[2/4] Analyzing income correlation...")
    try:
        demographics = load_demographics()
        all_results['income'] = analyze_income_correlation(results, demographics)
        print(f"  Spearman r = {all_results['income']['spearman_correlation']:.3f} (p={all_results['income']['p_value']:.4f})")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
        all_results['income'] = None

    # Spending patterns
    print("[3/4] Analyzing spending patterns...")
    all_results['spending'] = analyze_spending_patterns(results)
    print(f"  Pearson r = {all_results['spending']['pearson_correlation']:.3f} (p={all_results['spending']['p_value']:.2e})")

    # Time/observation trends
    print("[4/4] Analyzing time trends...")
    all_results['time'] = analyze_observation_effect(results)
    print(f"  Pearson r = {all_results['time']['pearson_correlation']:.3f} (p={all_results['time']['p_value']:.2e})")

    print("\n" + "=" * 60)
    print(" ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nVisualizations saved to: {OUTPUT_DIR}")

    return all_results


if __name__ == "__main__":
    results = run_extended_analysis()

    print("\n" + "=" * 60)
    print(" KEY FINDINGS")
    print("=" * 60)

    s = results['summary']
    print(f"\nDataset: {s['total_households']} households")
    print(f"Mean integrity score: {s['mean_aei']:.3f}")
    print(f"GARP-consistent: {s['consistent_count']} ({s['consistent_pct']:.1f}%)")
    print(f"Erratic (<0.7): {s['below_0.7']} ({s['below_0.7_pct']:.1f}%)")

    if results['income']:
        i = results['income']
        sig = "***" if i['p_value'] < 0.001 else "**" if i['p_value'] < 0.01 else "*" if i['p_value'] < 0.05 else ""
        print(f"\nIncome correlation: r={i['spearman_correlation']:.3f}{sig}")
        print(f"  {i['interpretation']}")

    sp = results['spending']
    sig = "***" if sp['p_value'] < 0.001 else "**" if sp['p_value'] < 0.01 else "*" if sp['p_value'] < 0.05 else ""
    print(f"\nSpending correlation: r={sp['pearson_correlation']:.3f}{sig}")
    print(f"  {sp['interpretation']}")

    t = results['time']
    sig = "***" if t['p_value'] < 0.001 else "**" if t['p_value'] < 0.01 else "*" if t['p_value'] < 0.05 else ""
    print(f"\nTime/experience correlation: r={t['pearson_correlation']:.3f}{sig}")
    print(f"  {t['interpretation']}")
