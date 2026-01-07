#!/usr/bin/env python3
"""New algorithms PyRevealed analysis on Dunnhumby data.

Showcase K: Bronars Power - Statistical significance of GARP tests
Showcase L: Homotheticity (HARP) - Tests if preferences scale with budget
Showcase M: Granular Efficiency (VEI) - Per-observation integrity scores
Showcase N: Income Invariance - Tests for income effects (quasilinearity)
Showcase O: Cross-Price Effects - Substitute/complement detection

2024 Survey Algorithms:
Showcase P: Smooth Preferences (Differentiable Rationality) - SARP + uniqueness
Showcase Q: Strict Consistency (Acyclical P) - Strict preference cycles only
Showcase R: Price Preferences (GAPP) - Consistent price preferences
"""

from __future__ import annotations

import sys
import random
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pyrevealed import (
    BehaviorLog,
    compute_test_power,
    validate_proportional_scaling,
    compute_granular_integrity,
    test_income_invariance,
    compute_cross_price_matrix,
    # 2024 Survey algorithms
    validate_smooth_preferences,
    validate_strict_consistency,
    validate_price_preferences,
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


def analyze_test_power(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 200,
    n_simulations: int = 100,
) -> dict:
    """
    Showcase K: Bronars' Power Index.

    Measures statistical significance of GARP test results.
    High power (>0.5) means passing GARP is meaningful.
    Low power means even random behavior would pass.
    """
    import matplotlib.pyplot as plt

    print("\n[Showcase K] Computing Bronars' Power Index...")

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    power_values = []
    significant_count = 0
    mean_random_aei_values = []

    for i, key in enumerate(sample_keys):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        try:
            log = sessions[key].behavior_log

            # Skip very small sessions
            if log.num_records < 5:
                continue

            result = compute_test_power(log, n_simulations=n_simulations)
            power_values.append(result.power_index)

            if result.is_significant:
                significant_count += 1

            if result.mean_integrity_random is not None:
                mean_random_aei_values.append(result.mean_integrity_random)

        except Exception as e:
            continue

    if not power_values:
        return {'error': 'No valid computations'}

    power_values = np.array(power_values)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Power distribution
    ax = axes[0]
    ax.hist(power_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Significance threshold')
    ax.axvline(np.mean(power_values), color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(power_values):.3f}')
    ax.set_xlabel("Bronars' Power Index")
    ax.set_ylabel('Number of Households')
    ax.set_title(f"Test Power Distribution (n={len(power_values)})\n"
                 f"Significant: {significant_count} ({100*significant_count/len(power_values):.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Random AEI distribution
    ax = axes[1]
    if mean_random_aei_values:
        ax.hist(mean_random_aei_values, bins=30, edgecolor='black', alpha=0.7, color='coral')
        ax.axvline(np.mean(mean_random_aei_values), color='black', linestyle='--',
                   label=f'Mean: {np.mean(mean_random_aei_values):.3f}')
        ax.set_xlabel('Mean Random AEI')
        ax.set_ylabel('Number of Households')
        ax.set_title('Mean Integrity of Random Behaviors\n(Lower = Test has more power)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_k_bronars_power.png', dpi=150)
    plt.close()

    return {
        'n_samples': len(power_values),
        'mean_power': round(float(np.mean(power_values)), 4),
        'median_power': round(float(np.median(power_values)), 4),
        'std_power': round(float(np.std(power_values)), 4),
        'pct_significant': round(100 * significant_count / len(power_values), 1),
        'mean_random_aei': round(float(np.mean(mean_random_aei_values)), 4) if mean_random_aei_values else None,
    }


def analyze_homotheticity(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 500,
) -> dict:
    """
    Showcase L: Homotheticity Test (HARP).

    Tests if preferences scale proportionally with budget.
    Homothetic preferences mean relative spending shares stay constant
    regardless of income level.
    """
    import matplotlib.pyplot as plt

    print("\n[Showcase L] Testing Homotheticity (HARP)...")

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    homothetic_count = 0
    max_cycle_products = []

    for i, key in enumerate(sample_keys):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        try:
            log = sessions[key].behavior_log

            result = validate_proportional_scaling(log)

            if result.is_consistent:
                homothetic_count += 1

            max_cycle_products.append(result.max_cycle_product)

        except Exception:
            continue

    if not max_cycle_products:
        return {'error': 'No valid computations'}

    total = len(max_cycle_products)
    max_cycle_products = np.array(max_cycle_products)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))

    # Filter to show violations (product > 1)
    violations = max_cycle_products[max_cycle_products > 1.0]
    if len(violations) > 0:
        ax.hist(violations, bins=30, edgecolor='black', alpha=0.7, color='tomato')
        ax.axvline(np.mean(violations), color='black', linestyle='--',
                   label=f'Mean violation: {np.mean(violations):.3f}')
        ax.set_xlabel('Max Cycle Product (1.0 = threshold)')
        ax.set_ylabel('Number of Households')
        ax.set_title(f'HARP Violation Severity (n={len(violations)} violators)\n'
                     f'Homothetic: {homothetic_count}/{total} ({100*homothetic_count/total:.1f}%)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, f'All {total} households are homothetic!',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(f'HARP Test Results\nHomothetic: {homothetic_count}/{total} (100%)')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_l_homotheticity.png', dpi=150)
    plt.close()

    return {
        'n_samples': total,
        'homothetic_count': homothetic_count,
        'pct_homothetic': round(100 * homothetic_count / total, 1),
        'mean_max_cycle_product': round(float(np.mean(max_cycle_products)), 4),
        'max_violation': round(float(np.max(max_cycle_products)), 4),
    }


def analyze_granular_efficiency(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 500,
) -> dict:
    """
    Showcase M: Varian's Efficiency Index (VEI).

    Per-observation efficiency scores (more granular than AEI).
    Identifies which specific observations are problematic.
    """
    import matplotlib.pyplot as plt

    print("\n[Showcase M] Computing Granular Efficiency (VEI)...")

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    mean_efficiencies = []
    min_efficiencies = []
    pct_problematic = []

    for i, key in enumerate(sample_keys):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        try:
            log = sessions[key].behavior_log

            result = compute_granular_integrity(log)

            mean_efficiencies.append(result.mean_efficiency)
            min_efficiencies.append(result.min_efficiency)

            # Fraction of problematic observations
            n_problematic = len(result.problematic_observations)
            pct = n_problematic / result.num_observations if result.num_observations > 0 else 0
            pct_problematic.append(pct)

        except Exception:
            continue

    if not mean_efficiencies:
        return {'error': 'No valid computations'}

    mean_efficiencies = np.array(mean_efficiencies)
    min_efficiencies = np.array(min_efficiencies)
    pct_problematic = np.array(pct_problematic)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Mean efficiency distribution
    ax = axes[0]
    ax.hist(mean_efficiencies, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(mean_efficiencies), color='red', linestyle='--',
               label=f'Mean: {np.mean(mean_efficiencies):.3f}')
    ax.set_xlabel('Mean Per-Observation Efficiency')
    ax.set_ylabel('Number of Households')
    ax.set_title('Mean VEI Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Min efficiency distribution
    ax = axes[1]
    ax.hist(min_efficiencies, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(np.mean(min_efficiencies), color='red', linestyle='--',
               label=f'Mean: {np.mean(min_efficiencies):.3f}')
    ax.set_xlabel('Minimum Per-Observation Efficiency')
    ax.set_ylabel('Number of Households')
    ax.set_title('Worst Observation Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter: mean vs min
    ax = axes[2]
    ax.scatter(mean_efficiencies, min_efficiencies, alpha=0.5, s=20, c='forestgreen')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Mean Efficiency')
    ax.set_ylabel('Min Efficiency')
    ax.set_title('Mean vs Worst Observation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_m_vei.png', dpi=150)
    plt.close()

    return {
        'n_samples': len(mean_efficiencies),
        'mean_efficiency': round(float(np.mean(mean_efficiencies)), 4),
        'mean_min_efficiency': round(float(np.mean(min_efficiencies)), 4),
        'std_efficiency': round(float(np.std(mean_efficiencies)), 4),
        'mean_pct_problematic': round(float(np.mean(pct_problematic)) * 100, 2),
    }


def analyze_income_invariance(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 500,
) -> dict:
    """
    Showcase N: Income Invariance (Quasilinearity).

    Tests if user behavior is invariant to income changes.
    Quasilinear preferences have no income effects.
    """
    import matplotlib.pyplot as plt

    print("\n[Showcase N] Testing Income Invariance (Quasilinearity)...")

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    quasilinear_count = 0
    violation_magnitudes = []

    for i, key in enumerate(sample_keys):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        try:
            log = sessions[key].behavior_log

            result = test_income_invariance(log)

            if result.is_quasilinear:
                quasilinear_count += 1

            violation_magnitudes.append(result.worst_violation_magnitude)

        except Exception:
            continue

    if not violation_magnitudes:
        return {'error': 'No valid computations'}

    total = len(violation_magnitudes)
    violation_magnitudes = np.array(violation_magnitudes)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))

    # Show violation magnitude distribution
    nonzero_violations = violation_magnitudes[violation_magnitudes > 0]
    if len(nonzero_violations) > 0:
        ax.hist(nonzero_violations, bins=30, edgecolor='black', alpha=0.7, color='purple')
        ax.axvline(np.mean(nonzero_violations), color='black', linestyle='--',
                   label=f'Mean: {np.mean(nonzero_violations):.3f}')
        ax.set_xlabel('Worst Violation Magnitude')
        ax.set_ylabel('Number of Households')
        ax.set_title(f'Income Effect Violations (n={len(nonzero_violations)} with violations)\n'
                     f'Quasilinear: {quasilinear_count}/{total} ({100*quasilinear_count/total:.1f}%)')
        ax.legend()
    else:
        ax.text(0.5, 0.5, f'All {total} households are quasilinear!',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(f'Quasilinearity Test Results\nQuasilinear: {quasilinear_count}/{total} (100%)')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_n_quasilinearity.png', dpi=150)
    plt.close()

    return {
        'n_samples': total,
        'quasilinear_count': quasilinear_count,
        'pct_quasilinear': round(100 * quasilinear_count / total, 1),
        'pct_with_income_effects': round(100 * (total - quasilinear_count) / total, 1),
        'mean_violation_magnitude': round(float(np.mean(violation_magnitudes)), 4),
    }


def analyze_cross_price_effects(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 200,
) -> dict:
    """
    Showcase O: Cross-Price Effects (Gross Substitutes).

    Determines substitute/complement relationships between products.
    Aggregates across households for consensus matrix.
    """
    import matplotlib.pyplot as plt

    print("\n[Showcase O] Analyzing Cross-Price Effects...")

    n_products = len(TOP_COMMODITIES)
    short_names = [COMMODITY_SHORT_NAMES.get(c, c[:6]) for c in TOP_COMMODITIES]

    # Count relationships across households
    substitute_counts = np.zeros((n_products, n_products))
    complement_counts = np.zeros((n_products, n_products))
    total_counts = np.zeros((n_products, n_products))

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))
    valid_count = 0

    for i, key in enumerate(sample_keys):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        try:
            log = sessions[key].behavior_log

            result = compute_cross_price_matrix(log)

            if result is None:
                continue

            valid_count += 1

            # Aggregate relationship classifications
            for g in range(n_products):
                for h in range(n_products):
                    if g == h:
                        continue

                    rel = result.relationship_matrix[g, h]
                    total_counts[g, h] += 1

                    if rel == 'substitutes':
                        substitute_counts[g, h] += 1
                    elif rel == 'complements':
                        complement_counts[g, h] += 1

        except Exception:
            continue

    if valid_count == 0:
        return {'error': 'No valid computations'}

    print(f"  Valid matrices from {valid_count}/{len(sample_keys)} households")

    # Compute net relationship scores
    # Positive = substitutes, Negative = complements
    total_counts = np.maximum(total_counts, 1)
    net_scores = (substitute_counts - complement_counts) / total_counts
    np.fill_diagonal(net_scores, 0)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(net_scores, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(n_products))
    ax.set_yticks(range(n_products))
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_yticklabels(short_names)

    for i in range(n_products):
        for j in range(n_products):
            val = net_scores[i, j]
            if i != j:
                color = 'white' if abs(val) > 0.3 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

    ax.set_title('Cross-Price Effect Matrix\n(Blue = Substitutes, Red = Complements)')
    plt.colorbar(im, label='Net Score (+ = substitutes, - = complements)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_o_cross_price.png', dpi=150)
    plt.close()

    # Extract key pairs
    key_pairs = []
    for i in range(n_products):
        for j in range(i + 1, n_products):
            score = net_scores[i, j]
            if abs(score) > 0.2:
                rel = "substitutes" if score > 0 else "complements"
                key_pairs.append({
                    'product_a': short_names[i],
                    'product_b': short_names[j],
                    'score': round(score, 3),
                    'relationship': rel,
                })

    key_pairs.sort(key=lambda x: abs(x['score']), reverse=True)

    return {
        'n_households': valid_count,
        'n_products': n_products,
        'key_pairs': key_pairs[:10],
        'mean_abs_score': round(float(np.mean(np.abs(net_scores[net_scores != 0]))), 4),
    }


# =============================================================================
# 2024 SURVEY ALGORITHMS
# =============================================================================


def analyze_smooth_preferences(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 500,
) -> dict:
    """
    Showcase P: Smooth Preferences (Differentiable Rationality).

    Tests if preferences are differentiable (smooth utility function).
    Requires both SARP (no indifference cycles) and price-quantity uniqueness.
    Based on Chiappori & Rochet (1987).
    """
    import matplotlib.pyplot as plt

    print("\n[Showcase P] Testing Smooth Preferences (Differentiable Rationality)...")

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    sarp_pass = 0
    uniqueness_pass = 0
    differentiable_pass = 0
    total = 0

    for i, key in enumerate(sample_keys):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        try:
            log = sessions[key].behavior_log

            result = validate_smooth_preferences(log)
            total += 1

            if result.satisfies_sarp:
                sarp_pass += 1
            if result.satisfies_uniqueness:
                uniqueness_pass += 1
            if result.is_differentiable:
                differentiable_pass += 1

        except Exception:
            continue

    if total == 0:
        return {'error': 'No valid computations'}

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['SARP\n(No indifference cycles)', 'Uniqueness\n(p≠p\' ⟹ x≠x\')', 'Differentiable\n(Both)']
    counts = [sarp_pass, uniqueness_pass, differentiable_pass]
    pcts = [100 * c / total for c in counts]
    colors = ['steelblue', 'coral', 'forestgreen']

    bars = ax.bar(categories, pcts, color=colors, edgecolor='black', alpha=0.7)

    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{pct:.1f}%\n({cnt})', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Percentage of Households')
    ax.set_title(f'Smooth Preferences Test (n={total})\n'
                 f'Differentiable = SARP ∧ Uniqueness')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_p_smooth_preferences.png', dpi=150)
    plt.close()

    return {
        'n_samples': total,
        'sarp_pass': sarp_pass,
        'pct_sarp': round(100 * sarp_pass / total, 1),
        'uniqueness_pass': uniqueness_pass,
        'pct_uniqueness': round(100 * uniqueness_pass / total, 1),
        'differentiable_pass': differentiable_pass,
        'pct_differentiable': round(100 * differentiable_pass / total, 1),
    }


def analyze_strict_consistency(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 500,
) -> dict:
    """
    Showcase Q: Strict Consistency (Acyclical P).

    Tests only strict preference cycles (more lenient than GARP).
    Passes if violations are only due to indifference (weak preferences).
    Based on Dziewulski (2023).
    """
    import matplotlib.pyplot as plt

    print("\n[Showcase Q] Testing Strict Consistency (Acyclical P)...")

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    acyclical_p_pass = 0
    garp_pass = 0
    approximately_rational = 0  # GARP fails but Acyclical P passes
    total = 0

    for i, key in enumerate(sample_keys):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        try:
            log = sessions[key].behavior_log

            result = validate_strict_consistency(log)
            total += 1

            if result.is_acyclical:
                acyclical_p_pass += 1
            if result.garp_consistent:
                garp_pass += 1
            if result.is_approximately_rational and not result.garp_consistent:
                approximately_rational += 1

        except Exception:
            continue

    if total == 0:
        return {'error': 'No valid computations'}

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Comparison bar chart
    ax = axes[0]
    categories = ['GARP\n(Full consistency)', 'Acyclical P\n(Strict cycles only)']
    counts = [garp_pass, acyclical_p_pass]
    pcts = [100 * c / total for c in counts]
    colors = ['steelblue', 'forestgreen']

    bars = ax.bar(categories, pcts, color=colors, edgecolor='black', alpha=0.7)

    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{pct:.1f}%\n({cnt})', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Percentage of Households')
    ax.set_title(f'GARP vs Acyclical P (n={total})')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    # Venn-style breakdown
    ax = axes[1]
    garp_only = garp_pass  # GARP ⟹ Acyclical P
    approx_only = approximately_rational
    neither = total - acyclical_p_pass

    categories = ['Fully Consistent\n(GARP pass)', 'Approximately Rational\n(GARP fail, Acyclical P pass)',
                  'Inconsistent\n(Both fail)']
    counts = [garp_only, approx_only, neither]
    pcts = [100 * c / total for c in counts]
    colors = ['forestgreen', 'gold', 'tomato']

    bars = ax.bar(categories, pcts, color=colors, edgecolor='black', alpha=0.7)

    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{pct:.1f}%\n({cnt})', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Percentage of Households')
    ax.set_title('Consistency Classification')
    ax.set_ylim(0, max(pcts) * 1.2 + 5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_q_strict_consistency.png', dpi=150)
    plt.close()

    return {
        'n_samples': total,
        'garp_pass': garp_pass,
        'pct_garp': round(100 * garp_pass / total, 1),
        'acyclical_p_pass': acyclical_p_pass,
        'pct_acyclical_p': round(100 * acyclical_p_pass / total, 1),
        'approximately_rational': approximately_rational,
        'pct_approximately_rational': round(100 * approximately_rational / total, 1),
    }


def analyze_price_preferences(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 500,
) -> dict:
    """
    Showcase R: Price Preferences (GAPP).

    Tests if user has consistent price preferences (dual of GARP).
    Tests whether users prefer situations where their desired items are cheaper.
    Based on Deb et al. (2022).
    """
    import matplotlib.pyplot as plt

    print("\n[Showcase R] Testing Price Preferences (GAPP)...")

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    gapp_pass = 0
    garp_pass = 0
    both_pass = 0
    total = 0

    for i, key in enumerate(sample_keys):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        try:
            log = sessions[key].behavior_log

            result = validate_price_preferences(log)
            total += 1

            gapp_ok = result.is_consistent
            garp_ok = result.garp_consistent

            if gapp_ok:
                gapp_pass += 1
            if garp_ok:
                garp_pass += 1
            if gapp_ok and garp_ok:
                both_pass += 1

        except Exception:
            continue

    if total == 0:
        return {'error': 'No valid computations'}

    # Compute overlap stats
    garp_only = garp_pass - both_pass
    gapp_only = gapp_pass - both_pass
    neither = total - garp_pass - gapp_only

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Comparison
    ax = axes[0]
    categories = ['GARP\n(Quantity consistency)', 'GAPP\n(Price consistency)', 'Both']
    counts = [garp_pass, gapp_pass, both_pass]
    pcts = [100 * c / total for c in counts]
    colors = ['steelblue', 'coral', 'forestgreen']

    bars = ax.bar(categories, pcts, color=colors, edgecolor='black', alpha=0.7)

    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{pct:.1f}%\n({cnt})', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Percentage of Households')
    ax.set_title(f'GARP vs GAPP Comparison (n={total})')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    # 4-way breakdown
    ax = axes[1]
    categories = ['Both Pass', 'GARP only', 'GAPP only', 'Neither']
    counts = [both_pass, garp_only, gapp_only, neither]
    pcts = [100 * c / total for c in counts]
    colors = ['forestgreen', 'steelblue', 'coral', 'gray']

    bars = ax.bar(categories, pcts, color=colors, edgecolor='black', alpha=0.7)

    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{pct:.1f}%\n({cnt})', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Percentage of Households')
    ax.set_title('GARP/GAPP Overlap')
    ax.set_ylim(0, max(pcts) * 1.2 + 5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_r_price_preferences.png', dpi=150)
    plt.close()

    return {
        'n_samples': total,
        'garp_pass': garp_pass,
        'pct_garp': round(100 * garp_pass / total, 1),
        'gapp_pass': gapp_pass,
        'pct_gapp': round(100 * gapp_pass / total, 1),
        'both_pass': both_pass,
        'pct_both': round(100 * both_pass / total, 1),
    }


def run_new_algorithms_analysis() -> dict:
    """Run all new algorithm analyses."""
    print("=" * 70)
    print(" NEW ALGORITHMS ANALYSIS")
    print("=" * 70)

    sessions = load_sessions()
    print(f"  Loaded {len(sessions)} sessions")

    results = {}

    # Showcase K
    results['bronars'] = analyze_test_power(sessions)

    # Showcase L
    results['harp'] = analyze_homotheticity(sessions)

    # Showcase M
    results['vei'] = analyze_granular_efficiency(sessions)

    # Showcase N
    results['quasilinearity'] = analyze_income_invariance(sessions)

    # Showcase O
    results['cross_price'] = analyze_cross_price_effects(sessions)

    # 2024 Survey Algorithms
    # Showcase P
    results['smooth_preferences'] = analyze_smooth_preferences(sessions)

    # Showcase Q
    results['strict_consistency'] = analyze_strict_consistency(sessions)

    # Showcase R
    results['price_preferences'] = analyze_price_preferences(sessions)

    print("\n" + "=" * 70)
    print(" NEW ALGORITHMS ANALYSIS COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_new_algorithms_analysis()

    print("\n" + "=" * 70)
    print(" KEY FINDINGS")
    print("=" * 70)

    if 'error' not in results.get('bronars', {}):
        print("\n[K] BRONARS' POWER INDEX (Test Significance)")
        b = results['bronars']
        print(f"  Households tested: {b['n_samples']}")
        print(f"  Mean power: {b['mean_power']:.3f}")
        print(f"  Significant (>0.5): {b['pct_significant']:.1f}%")
        if b['mean_random_aei']:
            print(f"  Mean random AEI: {b['mean_random_aei']:.3f}")

    if 'error' not in results.get('harp', {}):
        print("\n[L] HOMOTHETICITY (HARP)")
        h = results['harp']
        print(f"  Households tested: {h['n_samples']}")
        print(f"  Homothetic: {h['pct_homothetic']:.1f}%")
        print(f"  Mean max cycle product: {h['mean_max_cycle_product']:.4f}")

    if 'error' not in results.get('vei', {}):
        print("\n[M] GRANULAR EFFICIENCY (VEI)")
        v = results['vei']
        print(f"  Households tested: {v['n_samples']}")
        print(f"  Mean efficiency: {v['mean_efficiency']:.3f}")
        print(f"  Mean min efficiency: {v['mean_min_efficiency']:.3f}")
        print(f"  Mean problematic observations: {v['mean_pct_problematic']:.1f}%")

    if 'error' not in results.get('quasilinearity', {}):
        print("\n[N] INCOME INVARIANCE (Quasilinearity)")
        q = results['quasilinearity']
        print(f"  Households tested: {q['n_samples']}")
        print(f"  Quasilinear (no income effects): {q['pct_quasilinear']:.1f}%")
        print(f"  Has income effects: {q['pct_with_income_effects']:.1f}%")

    if 'error' not in results.get('cross_price', {}):
        print("\n[O] CROSS-PRICE EFFECTS")
        c = results['cross_price']
        print(f"  Households analyzed: {c['n_households']}")
        print(f"  Key relationships:")
        for pair in c['key_pairs'][:5]:
            print(f"    {pair['product_a']} & {pair['product_b']}: {pair['score']:.3f} ({pair['relationship']})")

    # 2024 Survey Algorithms
    if 'error' not in results.get('smooth_preferences', {}):
        print("\n[P] SMOOTH PREFERENCES (Differentiable Rationality)")
        sp = results['smooth_preferences']
        print(f"  Households tested: {sp['n_samples']}")
        print(f"  SARP pass: {sp['pct_sarp']:.1f}%")
        print(f"  Uniqueness pass: {sp['pct_uniqueness']:.1f}%")
        print(f"  Differentiable (both): {sp['pct_differentiable']:.1f}%")

    if 'error' not in results.get('strict_consistency', {}):
        print("\n[Q] STRICT CONSISTENCY (Acyclical P)")
        sc = results['strict_consistency']
        print(f"  Households tested: {sc['n_samples']}")
        print(f"  GARP pass: {sc['pct_garp']:.1f}%")
        print(f"  Acyclical P pass: {sc['pct_acyclical_p']:.1f}%")
        print(f"  Approximately rational (GARP fail, Acyclical P pass): {sc['pct_approximately_rational']:.1f}%")

    if 'error' not in results.get('price_preferences', {}):
        print("\n[R] PRICE PREFERENCES (GAPP)")
        pp = results['price_preferences']
        print(f"  Households tested: {pp['n_samples']}")
        print(f"  GARP pass: {pp['pct_garp']:.1f}%")
        print(f"  GAPP pass: {pp['pct_gapp']:.1f}%")
        print(f"  Both pass: {pp['pct_both']:.1f}%")

    print(f"\nVisualizations saved to: {OUTPUT_DIR}")
