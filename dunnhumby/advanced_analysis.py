#!/usr/bin/env python3
"""Advanced PyRevealed analysis on Dunnhumby data.

Showcase D: Complementarity Matrix - Product pair relationships
Showcase E: Mental Accounting - Separability between budget categories
Showcase F: Inflation Stress Test - Counterfactual demand under price shock
Showcase G: Structural Breaks - Rolling window MPI to detect preference shifts
"""

from __future__ import annotations

import sys
import time
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pyrevealed import (
    BehaviorLog,
    check_separability,
    compute_mpi,
    compute_aei,
    recover_utility,
    predict_demand,
)

from config import OUTPUT_DIR, TOP_COMMODITIES, COMMODITY_SHORT_NAMES
from session_builder import HouseholdData


def load_sessions() -> Dict[int, HouseholdData]:
    """Load pre-built sessions."""
    cache_file = OUTPUT_DIR.parent / "cache" / "sessions.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    from data_loader import load_filtered_data
    from price_oracle import get_master_price_grid
    from session_builder import build_all_sessions

    print("  Rebuilding sessions...")
    filtered_data = load_filtered_data(use_cache=True)
    price_grid = get_master_price_grid(filtered_data, use_cache=True)
    sessions = build_all_sessions(filtered_data, price_grid)

    with open(cache_file, 'wb') as f:
        pickle.dump(sessions, f)

    return sessions


def analyze_complementarity(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 300,
) -> dict:
    """
    Showcase D: Complementarity Matrix.

    For each product pair, compute correlation of quantity changes.
    Positive correlation = complements (bought together).
    Negative correlation = substitutes (one replaces other).
    """
    import matplotlib.pyplot as plt

    print("\n[Showcase D] Computing Complementarity Matrix...")

    n_products = len(TOP_COMMODITIES)
    short_names = [COMMODITY_SHORT_NAMES.get(c, c[:6]) for c in TOP_COMMODITIES]

    # Correlation matrix across households
    complement_matrix = np.zeros((n_products, n_products))
    sample_counts = np.zeros((n_products, n_products))

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    for key in sample_keys:
        log = sessions[key].behavior_log
        Q = log.action_vectors  # T x N

        if Q.shape[0] < 5:  # Need enough observations
            continue

        # Compute pairwise correlations within this household
        for i in range(n_products):
            for j in range(i + 1, n_products):
                q_i = Q[:, i]
                q_j = Q[:, j]

                # Skip if no variation
                if np.std(q_i) < 0.01 or np.std(q_j) < 0.01:
                    continue

                corr = np.corrcoef(q_i, q_j)[0, 1]
                if not np.isnan(corr):
                    complement_matrix[i, j] += corr
                    complement_matrix[j, i] += corr
                    sample_counts[i, j] += 1
                    sample_counts[j, i] += 1

    # Average correlations
    sample_counts = np.maximum(sample_counts, 1)  # Avoid div by zero
    complement_matrix /= sample_counts
    np.fill_diagonal(complement_matrix, 1.0)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(complement_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5)

    ax.set_xticks(range(n_products))
    ax.set_yticks(range(n_products))
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_yticklabels(short_names)

    for i in range(n_products):
        for j in range(n_products):
            val = complement_matrix[i, j]
            color = 'white' if abs(val) > 0.25 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

    ax.set_title('Product Complementarity Matrix\n(Blue=Substitutes, Red=Complements)')
    plt.colorbar(im, label='Correlation')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_d_complementarity.png', dpi=150)
    plt.close()

    # Extract key findings
    findings = []
    for i in range(n_products):
        for j in range(i + 1, n_products):
            corr = complement_matrix[i, j]
            if abs(corr) > 0.1:
                rel = "complements" if corr > 0 else "substitutes"
                findings.append({
                    'product_a': short_names[i],
                    'product_b': short_names[j],
                    'correlation': round(corr, 3),
                    'relationship': rel,
                })

    findings.sort(key=lambda x: abs(x['correlation']), reverse=True)

    return {
        'matrix': complement_matrix.tolist(),
        'product_names': short_names,
        'top_pairs': findings[:10],
    }


def analyze_mental_accounting(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 200,
) -> dict:
    """
    Showcase E: Mental Accounting (Separability Test).

    Test if households keep separate "mental budgets" for product categories.
    """
    import matplotlib.pyplot as plt

    print("\n[Showcase E] Testing Mental Accounting (Separability)...")

    # Define category groups (indices in TOP_COMMODITIES)
    # 0=Soda, 1=Milk, 2=Bread, 3=Cheese, 4=Chips, 5=Soup, 6=Yogurt, 7=Beef, 8=Pizza, 9=Lunchmeat
    categories = {
        'Dairy': [1, 3, 6],       # Milk, Cheese, Yogurt
        'Protein': [7, 9],        # Beef, Lunchmeat
        'Snacks': [0, 4, 8],      # Soda, Chips, Pizza
        'Staples': [2, 5],        # Bread, Soup
    }

    cat_names = list(categories.keys())
    n_cats = len(cat_names)

    # Test separability between each pair of categories
    separability_matrix = np.zeros((n_cats, n_cats))
    separable_counts = np.zeros((n_cats, n_cats))
    total_counts = np.zeros((n_cats, n_cats))

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    for key in sample_keys:
        log = sessions[key].behavior_log

        for i, cat_a in enumerate(cat_names):
            for j, cat_b in enumerate(cat_names):
                if i >= j:
                    continue

                group_a = categories[cat_a]
                group_b = categories[cat_b]

                try:
                    result = check_separability(log, group_a, group_b)
                    total_counts[i, j] += 1
                    total_counts[j, i] += 1
                    if result.is_separable:
                        separable_counts[i, j] += 1
                        separable_counts[j, i] += 1
                except Exception:
                    continue

    # Compute separability rates
    total_counts = np.maximum(total_counts, 1)
    separability_matrix = separable_counts / total_counts
    np.fill_diagonal(separability_matrix, 1.0)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(separability_matrix, cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_xticks(range(n_cats))
    ax.set_yticks(range(n_cats))
    ax.set_xticklabels(cat_names)
    ax.set_yticklabels(cat_names)

    for i in range(n_cats):
        for j in range(n_cats):
            val = separability_matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            pct = f'{val*100:.0f}%'
            ax.text(j, i, pct, ha='center', va='center', color=color, fontsize=14)

    ax.set_title('Category Separability (Mental Accounting)\n(Green=Separate Budgets, Red=Pooled Budget)')
    plt.colorbar(im, label='Separability Rate')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_e_mental_accounting.png', dpi=150)
    plt.close()

    # Summary stats
    findings = []
    for i, cat_a in enumerate(cat_names):
        for j, cat_b in enumerate(cat_names):
            if i < j:
                rate = separability_matrix[i, j]
                findings.append({
                    'category_a': cat_a,
                    'category_b': cat_b,
                    'separability_rate': round(rate, 3),
                    'separate_budgets': rate > 0.5,
                })

    return {
        'matrix': separability_matrix.tolist(),
        'categories': cat_names,
        'category_members': {k: [COMMODITY_SHORT_NAMES.get(TOP_COMMODITIES[i], '') for i in v]
                            for k, v in categories.items()},
        'findings': findings,
    }


def analyze_inflation_stress(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 100,
    inflation_rate: float = 0.20,
) -> dict:
    """
    Showcase F: Inflation Stress Test.

    Simulate price increase and predict demand changes using recovered utility.
    """
    import matplotlib.pyplot as plt

    print(f"\n[Showcase F] Inflation Stress Test ({inflation_rate*100:.0f}% price increase)...")

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    demand_reductions = []
    at_risk_households = []

    for i, key in enumerate(sample_keys):
        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        hh_data = sessions[key]
        log = hh_data.behavior_log

        # Recover utility
        try:
            utility_result = recover_utility(log)
            if not utility_result.success:
                continue
        except Exception:
            continue

        # Original demand (last observation)
        original_prices = log.cost_vectors[-1]
        original_qty = log.action_vectors[-1]
        original_total = np.sum(original_qty)
        budget = log.total_spend[-1]

        if original_total < 1:
            continue

        # Inflated prices
        new_prices = original_prices * (1 + inflation_rate)

        # Predict new demand
        try:
            new_qty = predict_demand(log, utility_result, new_prices, budget)
            if new_qty is None:
                continue
        except Exception:
            continue

        new_total = np.sum(new_qty)
        reduction = (original_total - new_total) / original_total

        demand_reductions.append(reduction)

        # "At risk" = demand drops by more than 50%
        if reduction > 0.5:
            at_risk_households.append({
                'household_key': key,
                'original_qty': original_total,
                'new_qty': new_total,
                'reduction_pct': round(reduction * 100, 1),
            })

    if not demand_reductions:
        print("  No valid predictions generated")
        return {'error': 'No valid predictions'}

    demand_reductions = np.array(demand_reductions)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of demand reduction
    ax = axes[0]
    ax.hist(demand_reductions * 100, bins=30, edgecolor='black', alpha=0.7, color='tomato')
    ax.axvline(np.mean(demand_reductions) * 100, color='black', linestyle='--',
               label=f'Mean: {np.mean(demand_reductions)*100:.1f}%')
    ax.axvline(np.median(demand_reductions) * 100, color='blue', linestyle='--',
               label=f'Median: {np.median(demand_reductions)*100:.1f}%')
    ax.set_xlabel('Demand Reduction (%)')
    ax.set_ylabel('Number of Households')
    ax.set_title(f'Demand Impact of {inflation_rate*100:.0f}% Price Increase')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # At-risk breakdown
    ax = axes[1]
    bins = [0, 10, 20, 30, 50, 100]
    labels = ['0-10%', '10-20%', '20-30%', '30-50%', '50%+']
    counts = []
    for i in range(len(bins) - 1):
        count = np.sum((demand_reductions * 100 >= bins[i]) & (demand_reductions * 100 < bins[i + 1]))
        counts.append(count)

    ax.bar(labels, counts, color=['green', 'yellowgreen', 'yellow', 'orange', 'red'], edgecolor='black')
    ax.set_xlabel('Demand Reduction Range')
    ax.set_ylabel('Number of Households')
    ax.set_title('Household Distribution by Impact Severity')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_f_inflation_stress.png', dpi=150)
    plt.close()

    return {
        'inflation_rate': inflation_rate,
        'n_households_tested': len(demand_reductions),
        'mean_demand_reduction': round(float(np.mean(demand_reductions)) * 100, 2),
        'median_demand_reduction': round(float(np.median(demand_reductions)) * 100, 2),
        'std_demand_reduction': round(float(np.std(demand_reductions)) * 100, 2),
        'pct_reduction_over_50': round(float(np.mean(demand_reductions > 0.5)) * 100, 1),
        'n_at_risk': len(at_risk_households),
        'at_risk_examples': at_risk_households[:5],
    }


def analyze_structural_breaks(
    sessions: Dict[int, HouseholdData],
    window_size: int = 10,
    min_observations: int = 50,
    sample_size: int = 50,
) -> dict:
    """
    Showcase G: Structural Breaks via Rolling MPI.

    Detect preference shifts by tracking MPI over time windows.
    """
    import matplotlib.pyplot as plt

    print(f"\n[Showcase G] Detecting Structural Breaks (rolling {window_size}-week MPI)...")

    # Filter to households with enough observations
    eligible = {k: v for k, v in sessions.items() if v.num_observations >= min_observations}
    print(f"  Households with {min_observations}+ observations: {len(eligible)}")

    if not eligible:
        return {'error': 'No households with enough observations'}

    keys = list(eligible.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    all_mpi_series = []
    break_detected = []

    for key in sample_keys:
        hh_data = eligible[key]
        log = hh_data.behavior_log

        # Split into windows
        windows = log.split_by_window(window_size)

        if len(windows) < 3:
            continue

        # Compute MPI for each window
        mpi_values = []
        for window_log in windows:
            try:
                result = compute_mpi(window_log)
                mpi_values.append(result.mpi_value)
            except Exception:
                mpi_values.append(np.nan)

        mpi_values = np.array(mpi_values)
        valid = ~np.isnan(mpi_values)

        if np.sum(valid) < 3:
            continue

        # Detect breaks: MPI spike > 2 std dev from mean
        mean_mpi = np.nanmean(mpi_values)
        std_mpi = np.nanstd(mpi_values)

        if std_mpi > 0.01:
            spikes = np.where(mpi_values > mean_mpi + 2 * std_mpi)[0]
            has_break = len(spikes) > 0
        else:
            has_break = False
            spikes = []

        all_mpi_series.append({
            'household_key': key,
            'mpi_values': mpi_values.tolist(),
            'mean_mpi': mean_mpi,
            'std_mpi': std_mpi,
            'break_detected': has_break,
            'break_windows': spikes.tolist() if has_break else [],
        })

        if has_break:
            break_detected.append(key)

    # Visualization: Example households
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot up to 4 example households with breaks
    examples_with_breaks = [s for s in all_mpi_series if s['break_detected']][:2]
    examples_without_breaks = [s for s in all_mpi_series if not s['break_detected']][:2]
    examples = examples_with_breaks + examples_without_breaks

    for idx, (ax, example) in enumerate(zip(axes.flat, examples)):
        mpi = example['mpi_values']
        x = range(len(mpi))

        ax.plot(x, mpi, 'b-o', markersize=4)
        ax.axhline(example['mean_mpi'], color='green', linestyle='--', alpha=0.7, label='Mean')
        ax.axhline(example['mean_mpi'] + 2 * example['std_mpi'], color='red', linestyle=':', alpha=0.7, label='+2σ')

        # Mark breaks
        for brk in example['break_windows']:
            ax.axvline(brk, color='red', alpha=0.5, linewidth=2)

        title = f"Household {example['household_key']}"
        if example['break_detected']:
            title += " (BREAK DETECTED)"
        ax.set_title(title)
        ax.set_xlabel(f'{window_size}-Week Window')
        ax.set_ylabel('MPI')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Rolling MPI Analysis ({window_size}-week windows)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_g_structural_breaks.png', dpi=150)
    plt.close()

    # Summary
    n_total = len(all_mpi_series)
    n_breaks = len(break_detected)

    return {
        'window_size': window_size,
        'n_households_analyzed': n_total,
        'n_breaks_detected': n_breaks,
        'break_rate': round(n_breaks / n_total * 100, 1) if n_total > 0 else 0,
        'examples': all_mpi_series[:5],
    }


def run_advanced_analysis() -> dict:
    """Run all advanced analyses."""
    print("=" * 70)
    print(" ADVANCED PYREVEALED ANALYSIS")
    print("=" * 70)

    sessions = load_sessions()
    print(f"  Loaded {len(sessions)} sessions")

    results = {}

    # Showcase D
    results['complementarity'] = analyze_complementarity(sessions)

    # Showcase E
    results['mental_accounting'] = analyze_mental_accounting(sessions)

    # Showcase F
    results['inflation_stress'] = analyze_inflation_stress(sessions)

    # Showcase G
    results['structural_breaks'] = analyze_structural_breaks(sessions)

    print("\n" + "=" * 70)
    print(" ADVANCED ANALYSIS COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_advanced_analysis()

    print("\n" + "=" * 70)
    print(" KEY FACTS")
    print("=" * 70)

    print("\n[D] COMPLEMENTARITY")
    for pair in results['complementarity']['top_pairs'][:5]:
        print(f"  {pair['product_a']} & {pair['product_b']}: r={pair['correlation']:.3f} ({pair['relationship']})")

    print("\n[E] MENTAL ACCOUNTING (SEPARABILITY)")
    for f in results['mental_accounting']['findings']:
        sep = "separate" if f['separate_budgets'] else "pooled"
        print(f"  {f['category_a']} vs {f['category_b']}: {f['separability_rate']*100:.0f}% ({sep})")

    if 'error' not in results['inflation_stress']:
        print(f"\n[F] INFLATION STRESS TEST ({results['inflation_stress']['inflation_rate']*100:.0f}% price shock)")
        print(f"  Households tested: {results['inflation_stress']['n_households_tested']}")
        print(f"  Mean demand reduction: {results['inflation_stress']['mean_demand_reduction']:.1f}%")
        print(f"  Households with >50% reduction: {results['inflation_stress']['pct_reduction_over_50']:.1f}%")

    if 'error' not in results['structural_breaks']:
        print(f"\n[G] STRUCTURAL BREAKS")
        print(f"  Households analyzed: {results['structural_breaks']['n_households_analyzed']}")
        print(f"  Breaks detected: {results['structural_breaks']['n_breaks_detected']} ({results['structural_breaks']['break_rate']:.1f}%)")

    print(f"\nVisualizations saved to: {OUTPUT_DIR}")
