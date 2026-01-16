#!/usr/bin/env python3
"""Comprehensive PyRevealed analysis on Dunnhumby data.

Uses ALL available PyRevealed methods:
1. compute_mpi() - Money Pump Index (exploitability)
2. compute_houtman_maks_index() - Minimum outliers for consistency
3. check_warp() - Weak Axiom violations
4. check_separability() - Product group independence
5. compute_cannibalization() - Cross-elasticity
6. BehavioralAuditor - Bot/shared account risk scores
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pyrevealed import (
    BehaviorLog,
    BehavioralAuditor,
    check_warp,
    compute_mpi,
    compute_houtman_maks_index,
    check_separability,
    compute_cannibalization,
)

from config import OUTPUT_DIR, TOP_COMMODITIES, COMMODITY_SHORT_NAMES
from session_builder import HouseholdData


def load_sessions() -> Dict[int, HouseholdData]:
    """Load pre-built sessions from previous analysis."""
    import pickle
    cache_file = OUTPUT_DIR.parent / "cache" / "sessions.pkl"

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # If no cache, rebuild
    from data_loader import load_filtered_data
    from price_oracle import get_master_price_grid
    from session_builder import build_all_sessions

    print("  Rebuilding sessions (no cache found)...")
    filtered_data = load_filtered_data(use_cache=True)
    price_grid = get_master_price_grid(filtered_data, use_cache=True)
    sessions = build_all_sessions(filtered_data, price_grid)

    # Cache for next time
    with open(cache_file, 'wb') as f:
        pickle.dump(sessions, f)

    return sessions


def analyze_mpi_distribution(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 500,
) -> dict:
    """
    Analyze Money Pump Index distribution.

    MPI measures how much money could be extracted from a user
    by exploiting their inconsistencies. Higher = more confused.
    """
    import matplotlib.pyplot as plt
    import random

    print("\n[MPI Analysis] Computing Money Pump Index...")

    # Sample households (MPI can be slow)
    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    mpi_values = []
    aei_values = []

    for i, key in enumerate(sample_keys):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        try:
            log = sessions[key].behavior_log
            result = compute_mpi(log)
            mpi_values.append(result.mpi_value)

            # Also get AEI for comparison
            from src.pyrevealed import compute_aei
            aei_result = compute_aei(log, tolerance=1e-4)
            aei_values.append(aei_result.efficiency_index)
        except Exception as e:
            continue

    mpi_values = np.array(mpi_values)
    aei_values = np.array(aei_values)

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MPI histogram
    ax = axes[0]
    ax.hist(mpi_values, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(np.mean(mpi_values), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(mpi_values):.3f}')
    ax.axvline(np.median(mpi_values), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {np.median(mpi_values):.3f}')
    ax.set_xlabel('Money Pump Index (MPI)')
    ax.set_ylabel('Number of Households')
    ax.set_title(f'Exploitability Distribution (n={len(mpi_values)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AEI vs MPI scatter
    ax = axes[1]
    ax.scatter(aei_values, mpi_values, alpha=0.5, s=20, c='steelblue')

    # Add correlation
    corr, p_val = stats.pearsonr(aei_values, mpi_values)
    ax.set_xlabel('Integrity Score (AEI)')
    ax.set_ylabel('Exploitability (MPI)')
    ax.set_title(f'Integrity vs Exploitability\n(Pearson r={corr:.3f}, p={p_val:.2e})')
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(aei_values, mpi_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(aei_values.min(), aei_values.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analysis_mpi_distribution.png', dpi=150)
    plt.close()

    return {
        'n_samples': len(mpi_values),
        'mean_mpi': float(np.mean(mpi_values)),
        'median_mpi': float(np.median(mpi_values)),
        'std_mpi': float(np.std(mpi_values)),
        'max_mpi': float(np.max(mpi_values)),
        'aei_mpi_correlation': float(corr),
        'aei_mpi_pvalue': float(p_val),
    }


def analyze_warp_violations(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 1000,
) -> dict:
    """
    Analyze WARP (Weak Axiom) violations.

    WARP checks for direct contradictions only (no transitivity).
    WARP violations ⊆ GARP violations.
    """
    import matplotlib.pyplot as plt
    import random

    print("\n[WARP Analysis] Checking Weak Axiom violations...")

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    warp_consistent = 0
    warp_violations = []

    for key in sample_keys:
        try:
            log = sessions[key].behavior_log
            warp_result = check_warp(log)
            if warp_result.is_consistent:
                warp_consistent += 1
            warp_violations.append(len(warp_result.violations))
        except Exception:
            continue

    total = len(warp_violations)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(warp_violations, bins=range(0, max(warp_violations) + 2),
            edgecolor='black', alpha=0.7, color='lightgreen')
    ax.axvline(np.mean(warp_violations), color='red', linestyle='--',
               label=f'Mean: {np.mean(warp_violations):.1f}')
    ax.set_xlabel('Number of WARP Violations')
    ax.set_ylabel('Number of Households')
    ax.set_title(f'WARP Violation Distribution\n({warp_consistent}/{total} = {100*warp_consistent/total:.1f}% consistent)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analysis_warp_violations.png', dpi=150)
    plt.close()

    return {
        'n_samples': total,
        'warp_consistent': warp_consistent,
        'warp_consistent_pct': 100 * warp_consistent / total,
        'mean_violations': float(np.mean(warp_violations)),
        'max_violations': int(max(warp_violations)),
    }


def analyze_separability(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 200,
) -> dict:
    """
    Analyze separability between product groups.

    Tests if certain product categories are chosen independently.
    """
    import matplotlib.pyplot as plt
    import random

    print("\n[Separability Analysis] Testing product group independence...")

    # Define product groups
    groups = {
        'Beverages': [0, 1],      # Soda, Milk
        'Snacks': [4, 8],         # Chips, Pizza
        'Protein': [7, 9],        # Beef, Lunchmeat
        'Dairy': [1, 3, 6],       # Milk, Cheese, Yogurt
        'Staples': [2, 5],        # Bread, Soup
    }

    # Test all pairs
    group_names = list(groups.keys())
    n_groups = len(group_names)
    separability_matrix = np.zeros((n_groups, n_groups))

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    for i, name_a in enumerate(group_names):
        for j, name_b in enumerate(group_names):
            if i >= j:
                continue

            group_a = groups[name_a]
            group_b = groups[name_b]

            # Skip if groups overlap
            if set(group_a) & set(group_b):
                separability_matrix[i, j] = np.nan
                separability_matrix[j, i] = np.nan
                continue

            sep_count = 0
            tested = 0

            for key in sample_keys:
                try:
                    log = sessions[key].behavior_log
                    result = check_separability(log, group_a, group_b)
                    if result.is_separable:
                        sep_count += 1
                    tested += 1
                except Exception:
                    continue

            if tested > 0:
                sep_rate = sep_count / tested
                separability_matrix[i, j] = sep_rate
                separability_matrix[j, i] = sep_rate

    # Fill diagonal with 1.0
    np.fill_diagonal(separability_matrix, 1.0)

    # Visualization - Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(separability_matrix, cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_xticks(range(n_groups))
    ax.set_yticks(range(n_groups))
    ax.set_xticklabels(group_names, rotation=45, ha='right')
    ax.set_yticklabels(group_names)

    # Add values
    for i in range(n_groups):
        for j in range(n_groups):
            val = separability_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

    ax.set_title('Product Group Separability\n(1.0 = fully independent, 0.0 = highly dependent)')
    plt.colorbar(im, label='Separability Rate')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analysis_separability_matrix.png', dpi=150)
    plt.close()

    return {
        'groups': groups,
        'separability_matrix': separability_matrix.tolist(),
        'group_names': group_names,
    }


def analyze_cannibalization(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 200,
) -> dict:
    """
    Analyze cannibalization (cross-elasticity) between products.

    Do purchases of one product reduce purchases of another?
    """
    import matplotlib.pyplot as plt
    import random

    print("\n[Cannibalization Analysis] Computing cross-elasticity...")

    # Test specific product pairs
    pairs = [
        (0, 1, 'Soda vs Milk'),       # Beverages
        (3, 9, 'Cheese vs Lunchmeat'), # Protein/dairy
        (4, 8, 'Chips vs Pizza'),      # Snacks
        (2, 5, 'Bread vs Soup'),       # Staples
        (6, 3, 'Yogurt vs Cheese'),    # Dairy
        (7, 9, 'Beef vs Lunchmeat'),   # Protein
    ]

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    results = []

    for idx_a, idx_b, name in pairs:
        cannib_values = []

        for key in sample_keys:
            try:
                log = sessions[key].behavior_log
                result = compute_cannibalization(log, [idx_a], [idx_b])
                cannib_values.append(result.cross_elasticity)
            except Exception:
                continue

        if cannib_values:
            results.append({
                'pair': name,
                'idx_a': idx_a,
                'idx_b': idx_b,
                'mean_cannib': np.mean(cannib_values),
                'std_cannib': np.std(cannib_values),
                'n_samples': len(cannib_values),
            })

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    pair_names = [r['pair'] for r in results]
    means = [r['mean_cannib'] for r in results]
    stds = [r['std_cannib'] for r in results]

    colors = ['red' if m < 0 else 'green' for m in means]
    bars = ax.barh(range(len(pair_names)), means, xerr=stds,
                   color=colors, alpha=0.7, capsize=5)

    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(range(len(pair_names)))
    ax.set_yticklabels(pair_names)
    ax.set_xlabel('Cross-Elasticity (negative = substitutes, positive = complements)')
    ax.set_title('Product Cannibalization Analysis')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analysis_cannibalization.png', dpi=150)
    plt.close()

    return {'pairs': results}


def analyze_with_auditor(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 500,
) -> dict:
    """
    Use BehavioralAuditor for comprehensive risk scores.
    """
    import matplotlib.pyplot as plt
    import random

    print("\n[Auditor Analysis] Computing risk scores...")

    auditor = BehavioralAuditor()

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    bot_risks = []
    shared_risks = []
    ux_risks = []
    integrity_scores = []

    for i, key in enumerate(sample_keys):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        try:
            log = sessions[key].behavior_log
            report = auditor.get_audit_report(log)

            bot_risks.append(report.bot_risk)
            shared_risks.append(report.shared_account_risk)
            ux_risks.append(report.ux_confusion_risk)
            integrity_scores.append(report.integrity_score)
        except Exception:
            continue

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Bot risk distribution
    ax = axes[0, 0]
    ax.hist(bot_risks, bins=50, edgecolor='black', alpha=0.7, color='red')
    ax.axvline(np.mean(bot_risks), color='black', linestyle='--',
               label=f'Mean: {np.mean(bot_risks):.3f}')
    ax.set_xlabel('Bot Risk Score')
    ax.set_ylabel('Count')
    ax.set_title(f'Bot Risk Distribution (>{0.5} flagged: {sum(1 for r in bot_risks if r > 0.5)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shared account risk
    ax = axes[0, 1]
    ax.hist(shared_risks, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.mean(shared_risks), color='black', linestyle='--',
               label=f'Mean: {np.mean(shared_risks):.3f}')
    ax.set_xlabel('Shared Account Risk Score')
    ax.set_ylabel('Count')
    ax.set_title(f'Shared Account Risk (>{0.5} flagged: {sum(1 for r in shared_risks if r > 0.5)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # UX confusion risk
    ax = axes[1, 0]
    ax.hist(ux_risks, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(np.mean(ux_risks), color='black', linestyle='--',
               label=f'Mean: {np.mean(ux_risks):.3f}')
    ax.set_xlabel('UX Confusion Risk Score')
    ax.set_ylabel('Count')
    ax.set_title(f'UX Confusion Risk (>{0.5} flagged: {sum(1 for r in ux_risks if r > 0.5)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Risk correlation scatter
    ax = axes[1, 1]
    ax.scatter(integrity_scores, bot_risks, alpha=0.5, s=20, label='Bot Risk')
    ax.scatter(integrity_scores, ux_risks, alpha=0.5, s=20, label='UX Risk')
    ax.set_xlabel('Integrity Score')
    ax.set_ylabel('Risk Score')
    ax.set_title('Integrity vs Risk Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analysis_auditor_risks.png', dpi=150)
    plt.close()

    return {
        'n_samples': len(bot_risks),
        'mean_bot_risk': float(np.mean(bot_risks)),
        'mean_shared_risk': float(np.mean(shared_risks)),
        'mean_ux_risk': float(np.mean(ux_risks)),
        'flagged_bots': sum(1 for r in bot_risks if r > 0.5),
        'flagged_shared': sum(1 for r in shared_risks if r > 0.5),
        'flagged_ux': sum(1 for r in ux_risks if r > 0.5),
    }


def run_comprehensive_analysis() -> dict:
    """Run all analyses."""
    print("=" * 70)
    print(" COMPREHENSIVE PYREVEALED ANALYSIS")
    print("=" * 70)

    # Load sessions
    print("\nLoading sessions...")
    sessions = load_sessions()
    print(f"  Loaded {len(sessions)} household sessions")

    all_results = {}

    # 1. MPI Analysis
    all_results['mpi'] = analyze_mpi_distribution(sessions, sample_size=500)
    print(f"  MPI: mean={all_results['mpi']['mean_mpi']:.4f}")

    # 2. WARP Analysis
    all_results['warp'] = analyze_warp_violations(sessions, sample_size=1000)
    print(f"  WARP consistent: {all_results['warp']['warp_consistent_pct']:.1f}%")

    # 3. Separability Analysis
    all_results['separability'] = analyze_separability(sessions, sample_size=200)
    print(f"  Separability analysis complete")

    # 4. Cannibalization Analysis
    all_results['cannibalization'] = analyze_cannibalization(sessions, sample_size=200)
    print(f"  Cannibalization analysis complete")

    # 5. Auditor Analysis
    all_results['auditor'] = analyze_with_auditor(sessions, sample_size=500)
    print(f"  Auditor: {all_results['auditor']['flagged_bots']} potential bots flagged")

    print("\n" + "=" * 70)
    print(" COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = run_comprehensive_analysis()

    print("\n" + "=" * 70)
    print(" KEY FINDINGS")
    print("=" * 70)

    print(f"\n1. EXPLOITABILITY (MPI)")
    print(f"   Mean MPI: {results['mpi']['mean_mpi']:.4f}")
    print(f"   Max MPI: {results['mpi']['max_mpi']:.4f}")
    print(f"   AEI-MPI correlation: r={results['mpi']['aei_mpi_correlation']:.3f}")

    print(f"\n2. WARP CONSISTENCY")
    print(f"   WARP-consistent: {results['warp']['warp_consistent_pct']:.1f}%")
    print(f"   (vs GARP-consistent: ~4.5%)")

    print(f"\n3. RISK SCORES (BehavioralAuditor)")
    print(f"   Mean bot risk: {results['auditor']['mean_bot_risk']:.3f}")
    print(f"   Flagged as bots: {results['auditor']['flagged_bots']}")
    print(f"   Flagged shared accounts: {results['auditor']['flagged_shared']}")
    print(f"   Flagged UX confusion: {results['auditor']['flagged_ux']}")

    print(f"\n4. PRODUCT RELATIONSHIPS")
    for pair in results['cannibalization']['pairs']:
        direction = "substitutes" if pair['mean_cannib'] < 0 else "complements"
        print(f"   {pair['pair']}: {pair['mean_cannib']:.3f} ({direction})")

    print(f"\nVisualizations saved to: {OUTPUT_DIR}")
