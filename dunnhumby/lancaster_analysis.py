#!/usr/bin/env python3
"""Lancaster Characteristics Model analysis on Dunnhumby data.

Showcase P: Lancaster Model - Nutritional Characteristics Analysis

This module transforms product-space grocery behavior into characteristics-space
using nutritional attributes (Protein, Carbs, Fat, Sodium). Key analyses:

1. "Rationality Rescue": Do households become more consistent when analyzed
   at the nutrient level rather than product level?

2. Shadow Prices: What are the implied valuations for each nutrient?

3. Spend Shares: What fraction of grocery spending goes to each nutrient?
"""

from __future__ import annotations

import sys
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pyrevealed import (
    BehaviorLog,
    LancasterLog,
    validate_consistency,
    compute_integrity_score,
)

from config import OUTPUT_DIR, TOP_COMMODITIES, COMMODITY_SHORT_NAMES

# Try to import session builder types
try:
    from session_builder import HouseholdData
except ImportError:
    HouseholdData = None


# =============================================================================
# NUTRITIONAL ATTRIBUTE MATRIX
# =============================================================================

# Characteristics: Protein (g), Carbs (g), Fat (g), Sodium (mg)
# Values are per typical serving size for each commodity category

CHARACTERISTIC_NAMES = ["Protein", "Carbs", "Fat", "Sodium"]

# Order matches TOP_COMMODITIES in config.py:
# ["SOFT DRINKS", "FLUID MILK PRODUCTS", "BAKED BREAD/BUNS/ROLLS", "CHEESE",
#  "BAG SNACKS", "SOUP", "YOGURT", "BEEF", "FROZEN PIZZA", "LUNCHMEAT"]

ATTRIBUTE_MATRIX = np.array([
    # Protein, Carbs, Fat, Sodium (per typical serving)
    [0.0,   39.0,   0.0,   40.0],   # Soda (12 oz can)
    [8.0,   12.0,   5.0,  120.0],   # Milk (1 cup)
    [4.0,   24.0,   1.0,  230.0],   # Bread (2 slices)
    [7.0,    1.0,   9.0,  180.0],   # Cheese (1 oz)
    [2.0,   15.0,  10.0,  180.0],   # Chips (1 oz)
    [3.0,   10.0,   2.0,  800.0],   # Soup (1 cup)
    [5.0,   15.0,   2.0,   70.0],   # Yogurt (6 oz)
    [26.0,   0.0,  15.0,   75.0],   # Beef (3 oz cooked)
    [12.0,  35.0,  10.0,  700.0],   # Pizza (1 slice)
    [10.0,   2.0,   4.0,  500.0],   # Lunchmeat (2 oz)
], dtype=np.float64)


def load_sessions() -> Dict[int, any]:
    """Load pre-built sessions from cache."""
    cache_file = OUTPUT_DIR.parent / "cache" / "sessions.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Rebuild if not cached
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


def analyze_lancaster_model(
    sessions: Dict[int, any],
    sample_size: int | None = None,
) -> dict:
    """
    Showcase P: Lancaster Characteristics Model.

    Transforms product-space behavior to nutritional characteristics-space
    and compares consistency metrics.

    Args:
        sessions: Dictionary mapping household_key to HouseholdData
        sample_size: Optional limit on households to analyze (None = all)

    Returns:
        Dictionary with analysis results
    """
    import matplotlib.pyplot as plt

    print("\n[Showcase P] Lancaster Characteristics Model (Nutritional Analysis)")
    print("=" * 70)

    keys = list(sessions.keys())
    if sample_size and sample_size < len(keys):
        import random
        random.seed(42)
        keys = random.sample(keys, sample_size)

    print(f"  Analyzing {len(keys)} households...")
    print(f"  Characteristics: {CHARACTERISTIC_NAMES}")

    # Storage for results
    product_integrity = []
    characteristic_integrity = []
    product_consistent = []
    characteristic_consistent = []
    shadow_prices_all = []
    spend_shares_all = []
    rationality_rescued = 0
    rationality_lost = 0

    for i, key in enumerate(keys):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(keys)}...")

        try:
            log = sessions[key].behavior_log

            # Skip very small sessions
            if log.num_records < 5:
                continue

            # Product-space analysis
            prod_result = compute_integrity_score(log, tolerance=1e-4)
            prod_consistent_flag = validate_consistency(log).is_consistent

            # Transform to characteristics space
            lancaster_log = LancasterLog(
                cost_vectors=log.cost_vectors,
                action_vectors=log.action_vectors,
                attribute_matrix=ATTRIBUTE_MATRIX,
                metadata={"characteristic_names": CHARACTERISTIC_NAMES},
            )

            # Characteristics-space analysis
            char_log = lancaster_log.behavior_log
            char_result = compute_integrity_score(char_log, tolerance=1e-4)
            char_consistent_flag = validate_consistency(char_log).is_consistent

            # Store results
            product_integrity.append(prod_result.efficiency_index)
            characteristic_integrity.append(char_result.efficiency_index)
            product_consistent.append(prod_consistent_flag)
            characteristic_consistent.append(char_consistent_flag)

            # Track rationality changes
            if not prod_consistent_flag and char_consistent_flag:
                rationality_rescued += 1
            elif prod_consistent_flag and not char_consistent_flag:
                rationality_lost += 1

            # Collect shadow prices and spend shares
            shadow_prices_all.append(lancaster_log.shadow_prices.mean(axis=0))
            report = lancaster_log.valuation_report()
            spend_shares_all.append(report.spend_shares)

        except Exception as e:
            continue

    if not product_integrity:
        return {'error': 'No valid computations'}

    # Convert to arrays
    product_integrity = np.array(product_integrity)
    characteristic_integrity = np.array(characteristic_integrity)
    product_consistent = np.array(product_consistent)
    characteristic_consistent = np.array(characteristic_consistent)
    shadow_prices_all = np.array(shadow_prices_all)
    spend_shares_all = np.array(spend_shares_all)

    n_total = len(product_integrity)

    # Compute statistics
    improved_integrity = characteristic_integrity > product_integrity + 0.01
    decreased_integrity = characteristic_integrity < product_integrity - 0.01

    mean_shadow_prices = np.mean(shadow_prices_all, axis=0)
    std_shadow_prices = np.std(shadow_prices_all, axis=0)
    mean_spend_shares = np.mean(spend_shares_all, axis=0)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Integrity comparison scatter
    ax = axes[0, 0]
    ax.scatter(product_integrity, characteristic_integrity, alpha=0.3, s=10, c='steelblue')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x (no change)')
    ax.set_xlabel('Product-Space Integrity (AEI)')
    ax.set_ylabel('Characteristics-Space Integrity (AEI)')
    ax.set_title(f'Integrity Comparison (n={n_total})\n'
                 f'Improved: {np.sum(improved_integrity)} ({100*np.mean(improved_integrity):.1f}%) | '
                 f'Decreased: {np.sum(decreased_integrity)} ({100*np.mean(decreased_integrity):.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    # 2. Integrity difference histogram
    ax = axes[0, 1]
    integrity_diff = characteristic_integrity - product_integrity
    ax.hist(integrity_diff, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='No change')
    ax.axvline(np.mean(integrity_diff), color='red', linestyle='--',
               label=f'Mean: {np.mean(integrity_diff):+.4f}')
    ax.set_xlabel('Integrity Change (Characteristics - Product)')
    ax.set_ylabel('Number of Households')
    ax.set_title('Distribution of Integrity Changes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Shadow prices bar chart
    ax = axes[1, 0]
    x = np.arange(len(CHARACTERISTIC_NAMES))
    bars = ax.bar(x, mean_shadow_prices, yerr=std_shadow_prices, capsize=5,
                  color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CHARACTERISTIC_NAMES)
    ax.set_ylabel('Shadow Price ($ per unit)')
    ax.set_title('Mean Shadow Prices (Implied Valuations)\n'
                 'Error bars show standard deviation')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, mean_shadow_prices):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'${val:.4f}', ha='center', va='bottom', fontsize=9)

    # 4. Spend shares pie chart
    ax = axes[1, 1]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    wedges, texts, autotexts = ax.pie(
        mean_spend_shares,
        labels=CHARACTERISTIC_NAMES,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
    )
    ax.set_title('Mean Spend Shares by Nutrient')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_p_lancaster.png', dpi=150)
    plt.close()

    # Additional: Rationality rescue visualization
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Perfect\n(Both)', 'Rescued\n(Char only)', 'Lost\n(Prod only)', 'Neither']
    both_consistent = np.sum(product_consistent & characteristic_consistent)
    neither_consistent = np.sum(~product_consistent & ~characteristic_consistent) - rationality_rescued
    counts = [both_consistent, rationality_rescued, rationality_lost, max(0, neither_consistent)]
    colors = ['#27ae60', '#2980b9', '#c0392b', '#7f8c8d']

    bars = ax.bar(categories, counts, color=colors, edgecolor='black')
    ax.set_ylabel('Number of Households')
    ax.set_title('Rationality Rescue: Product vs Characteristics Space\n'
                 f'"Rescued" = Consistent in characteristics but not products')

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{count}\n({100*count/n_total:.1f}%)', ha='center', va='bottom')

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_p_rationality_rescue.png', dpi=150)
    plt.close()

    # Correlation analysis
    corr, pval = stats.pearsonr(product_integrity, characteristic_integrity)

    return {
        'n_households': n_total,
        'product_mean_integrity': round(float(np.mean(product_integrity)), 4),
        'characteristic_mean_integrity': round(float(np.mean(characteristic_integrity)), 4),
        'mean_integrity_change': round(float(np.mean(integrity_diff)), 4),
        'pct_improved': round(100 * np.mean(improved_integrity), 1),
        'pct_decreased': round(100 * np.mean(decreased_integrity), 1),
        'product_consistent_count': int(np.sum(product_consistent)),
        'characteristic_consistent_count': int(np.sum(characteristic_consistent)),
        'rationality_rescued': rationality_rescued,
        'rationality_rescued_pct': round(100 * rationality_rescued / n_total, 1),
        'rationality_lost': rationality_lost,
        'correlation': round(corr, 4),
        'shadow_prices': {
            name: round(float(mean_shadow_prices[i]), 4)
            for i, name in enumerate(CHARACTERISTIC_NAMES)
        },
        'shadow_prices_std': {
            name: round(float(std_shadow_prices[i]), 4)
            for i, name in enumerate(CHARACTERISTIC_NAMES)
        },
        'spend_shares': {
            name: round(float(mean_spend_shares[i]) * 100, 1)
            for i, name in enumerate(CHARACTERISTIC_NAMES)
        },
    }


def run_lancaster_analysis() -> dict:
    """Run the Lancaster Characteristics Model analysis."""
    print("=" * 70)
    print(" LANCASTER CHARACTERISTICS MODEL ANALYSIS")
    print("=" * 70)

    sessions = load_sessions()
    print(f"  Loaded {len(sessions)} sessions")

    results = analyze_lancaster_model(sessions)

    print("\n" + "=" * 70)
    print(" LANCASTER ANALYSIS COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_lancaster_analysis()

    if 'error' not in results:
        print("\n" + "=" * 70)
        print(" KEY FINDINGS")
        print("=" * 70)

        print(f"\n[P] LANCASTER CHARACTERISTICS MODEL")
        print(f"  Households analyzed: {results['n_households']}")

        print(f"\n  CONSISTENCY COMPARISON:")
        print(f"    Product-space mean integrity: {results['product_mean_integrity']:.3f}")
        print(f"    Characteristics-space mean integrity: {results['characteristic_mean_integrity']:.3f}")
        print(f"    Mean change: {results['mean_integrity_change']:+.4f}")
        print(f"    Improved (>1% gain): {results['pct_improved']:.1f}%")
        print(f"    Decreased (>1% loss): {results['pct_decreased']:.1f}%")
        print(f"    Correlation: {results['correlation']:.3f}")

        print(f"\n  RATIONALITY RESCUE:")
        print(f"    Perfect in both: {results['product_consistent_count']} product, "
              f"{results['characteristic_consistent_count']} characteristic")
        print(f"    Rescued (char only): {results['rationality_rescued']} "
              f"({results['rationality_rescued_pct']:.1f}%)")
        print(f"    Lost (prod only): {results['rationality_lost']}")

        print(f"\n  SHADOW PRICES ($/unit):")
        for name, price in results['shadow_prices'].items():
            std = results['shadow_prices_std'][name]
            unit = "g" if name != "Sodium" else "mg"
            print(f"    {name}: ${price:.4f}/{unit} (std: ${std:.4f})")

        print(f"\n  SPEND SHARES:")
        for name, share in results['spend_shares'].items():
            print(f"    {name}: {share:.1f}%")

        print(f"\nVisualizations saved to: {OUTPUT_DIR}")
