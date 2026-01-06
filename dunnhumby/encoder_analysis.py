#!/usr/bin/env python3
"""Encoder-based PyRevealed analysis on Dunnhumby data.

Showcase H: Preference Encoder Features - Extract ML features for household clustering
Showcase I: Auto-Discovered Product Groups - Data-driven separability groupings
Showcase J: Houtman-Maks Robustness - Minimum outliers to remove for consistency
"""

from __future__ import annotations

import sys
import random
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pyrevealed import (
    BehaviorLog,
    PreferenceEncoder,
    discover_independent_groups,
    compute_minimal_outlier_fraction,
    compute_aei,
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


def analyze_preference_features(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 300,
    n_clusters: int = 4,
) -> dict:
    """
    Showcase H: Preference Encoder Features.

    Extract latent values and marginal weights from PreferenceEncoder,
    then cluster households by their preference profiles.
    """
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    print("\n[Showcase H] Extracting Preference Encoder Features...")

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    # Extract features for each household
    features = []
    household_ids = []
    fit_success_count = 0

    for i, key in enumerate(sample_keys):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        try:
            log = sessions[key].behavior_log

            encoder = PreferenceEncoder()
            encoder.fit(log)

            if not encoder.is_fitted:
                continue

            fit_success_count += 1

            # Extract aggregate features
            latent_values = encoder.extract_latent_values()
            marginal_weights = encoder.extract_marginal_weights()

            # Household-level features
            mean_latent = np.mean(latent_values)
            std_latent = np.std(latent_values)
            mean_marginal = encoder.mean_marginal_weight or 0.0

            # Also compute AEI for comparison
            aei_result = compute_aei(log, tolerance=1e-4)
            aei = aei_result.efficiency_index

            features.append({
                'household_key': key,
                'mean_latent': mean_latent,
                'std_latent': std_latent,
                'mean_marginal': mean_marginal,
                'aei': aei,
                'n_observations': log.num_records,
            })
            household_ids.append(key)

        except Exception as e:
            continue

    if len(features) < n_clusters:
        print(f"  Only {len(features)} households fitted, need at least {n_clusters}")
        return {'error': 'Not enough fitted households'}

    print(f"  Successfully fitted {fit_success_count}/{len(sample_keys)} households")

    # Convert to arrays for clustering
    df = pd.DataFrame(features)
    X = df[['mean_latent', 'std_latent', 'mean_marginal']].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['cluster'] = clusters

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter: mean_latent vs mean_marginal colored by cluster
    ax = axes[0]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for c in range(n_clusters):
        mask = df['cluster'] == c
        ax.scatter(
            df.loc[mask, 'mean_latent'],
            df.loc[mask, 'mean_marginal'],
            c=colors[c % len(colors)],
            alpha=0.6,
            s=40,
            label=f'Cluster {c+1} (n={mask.sum()})'
        )
    ax.set_xlabel('Mean Latent Value')
    ax.set_ylabel('Mean Marginal Weight (Price Sensitivity)')
    ax.set_title('Household Preference Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cluster profiles bar chart
    ax = axes[1]
    cluster_profiles = df.groupby('cluster')[['mean_latent', 'std_latent', 'mean_marginal', 'aei']].mean()

    x = np.arange(n_clusters)
    width = 0.2
    ax.bar(x - 1.5*width, cluster_profiles['mean_latent'], width, label='Mean Latent', color='steelblue')
    ax.bar(x - 0.5*width, cluster_profiles['std_latent'], width, label='Std Latent', color='coral')
    ax.bar(x + 0.5*width, cluster_profiles['mean_marginal'], width, label='Mean Marginal', color='forestgreen')
    ax.bar(x + 1.5*width, cluster_profiles['aei'], width, label='AEI', color='purple')

    ax.set_xlabel('Cluster')
    ax.set_ylabel('Feature Value (scaled)')
    ax.set_title('Cluster Feature Profiles')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cluster {i+1}' for i in range(n_clusters)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_h_preference_clusters.png', dpi=150)
    plt.close()

    # Summary statistics per cluster
    cluster_summary = []
    for c in range(n_clusters):
        mask = df['cluster'] == c
        cluster_df = df[mask]
        cluster_summary.append({
            'cluster': c + 1,
            'n_households': int(mask.sum()),
            'mean_latent': round(cluster_df['mean_latent'].mean(), 4),
            'mean_marginal': round(cluster_df['mean_marginal'].mean(), 4),
            'mean_aei': round(cluster_df['aei'].mean(), 3),
            'mean_observations': round(cluster_df['n_observations'].mean(), 1),
        })

    return {
        'n_fitted': len(features),
        'n_clusters': n_clusters,
        'cluster_summary': cluster_summary,
        'feature_means': {
            'mean_latent': round(df['mean_latent'].mean(), 4),
            'std_latent': round(df['std_latent'].mean(), 4),
            'mean_marginal': round(df['mean_marginal'].mean(), 4),
        },
    }


def analyze_auto_groups(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 200,
    max_groups: int = 3,
) -> dict:
    """
    Showcase I: Auto-Discovered Product Groups.

    Use discover_independent_groups() to find natural product clusters
    based on household behavior patterns.
    """
    import matplotlib.pyplot as plt

    print("\n[Showcase I] Auto-Discovering Product Groups...")

    n_products = len(TOP_COMMODITIES)
    short_names = [COMMODITY_SHORT_NAMES.get(c, c[:6]) for c in TOP_COMMODITIES]

    # Count how often each pair of products is in the same group
    co_cluster_matrix = np.zeros((n_products, n_products))
    total_counts = np.zeros((n_products, n_products))

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))
    valid_count = 0

    for i, key in enumerate(sample_keys):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        try:
            log = sessions[key].behavior_log

            # Auto-discover groups
            groups = discover_independent_groups(log, max_groups=max_groups)

            if not groups:
                continue

            valid_count += 1

            # Update co-clustering counts
            for group in groups:
                for p1 in group:
                    for p2 in group:
                        if p1 < n_products and p2 < n_products:
                            co_cluster_matrix[p1, p2] += 1

            # Count total observations
            for p1 in range(n_products):
                for p2 in range(n_products):
                    total_counts[p1, p2] += 1

        except Exception as e:
            continue

    print(f"  Valid groupings from {valid_count}/{len(sample_keys)} households")

    # Compute co-clustering rates
    total_counts = np.maximum(total_counts, 1)
    co_cluster_rate = co_cluster_matrix / total_counts

    # Compare to manual groupings
    manual_groups = {
        'Dairy': [1, 3, 6],       # Milk, Cheese, Yogurt
        'Protein': [7, 9],        # Beef, Lunchmeat
        'Snacks': [0, 4, 8],      # Soda, Chips, Pizza
        'Staples': [2, 5],        # Bread, Soup
    }

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Heatmap of co-clustering
    ax = axes[0]
    im = ax.imshow(co_cluster_rate, cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_xticks(range(n_products))
    ax.set_yticks(range(n_products))
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_yticklabels(short_names)

    for i in range(n_products):
        for j in range(n_products):
            val = co_cluster_rate[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

    ax.set_title('Product Co-Clustering Frequency\n(How often products grouped together)')
    plt.colorbar(im, ax=ax, label='Co-cluster Rate')

    # Manual groups comparison
    ax = axes[1]
    # Show which manual groups agree with auto-discovery
    group_names = list(manual_groups.keys())
    agreement_scores = []

    for gname, indices in manual_groups.items():
        if len(indices) < 2:
            agreement_scores.append(1.0)
            continue
        # Average co-clustering rate within manual group
        rates = []
        for i in indices:
            for j in indices:
                if i != j:
                    rates.append(co_cluster_rate[i, j])
        agreement_scores.append(np.mean(rates) if rates else 0)

    colors = ['green' if s > 0.5 else 'red' for s in agreement_scores]
    bars = ax.bar(group_names, agreement_scores, color=colors, edgecolor='black')

    ax.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_xlabel('Manual Category')
    ax.set_ylabel('Internal Co-Clustering Rate')
    ax.set_title('Manual Group vs Auto-Discovery Agreement\n(Green=Confirmed, Red=Not Confirmed)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, score in zip(bars, agreement_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_i_auto_groups.png', dpi=150)
    plt.close()

    # Find consensus groups (products with >70% co-clustering)
    consensus_pairs = []
    for i in range(n_products):
        for j in range(i + 1, n_products):
            if co_cluster_rate[i, j] > 0.7:
                consensus_pairs.append({
                    'product_a': short_names[i],
                    'product_b': short_names[j],
                    'co_cluster_rate': round(co_cluster_rate[i, j], 3),
                })

    return {
        'n_households': valid_count,
        'consensus_pairs': consensus_pairs,
        'manual_agreement': {g: round(s, 3) for g, s in zip(group_names, agreement_scores)},
        'confirmed_manual_groups': [g for g, s in zip(group_names, agreement_scores) if s > 0.5],
    }


def analyze_houtman_maks(
    sessions: Dict[int, HouseholdData],
    sample_size: int = 500,
) -> dict:
    """
    Showcase J: Houtman-Maks Robustness Index.

    Compute the minimum fraction of observations that need to be removed
    to make behavior consistent. Lower = more robust rationality.
    """
    import matplotlib.pyplot as plt

    print("\n[Showcase J] Computing Houtman-Maks Index...")

    keys = list(sessions.keys())
    sample_keys = random.sample(keys, min(sample_size, len(keys)))

    hm_values = []
    aei_values = []
    household_data = []

    for i, key in enumerate(sample_keys):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sample_keys)}...")

        try:
            log = sessions[key].behavior_log

            # Compute Houtman-Maks index
            hm_fraction, removed_indices = compute_minimal_outlier_fraction(log)

            # Compute AEI for comparison
            aei_result = compute_aei(log, tolerance=1e-4)
            aei = aei_result.efficiency_index

            hm_values.append(hm_fraction)
            aei_values.append(aei)

            household_data.append({
                'household_key': key,
                'hm_fraction': hm_fraction,
                'n_removed': len(removed_indices),
                'n_total': log.num_records,
                'aei': aei,
                'removed_indices': removed_indices[:5],  # First 5 for reference
            })

        except Exception as e:
            continue

    if not hm_values:
        return {'error': 'No valid computations'}

    hm_values = np.array(hm_values)
    aei_values = np.array(aei_values)

    print(f"  Computed HMI for {len(hm_values)} households")

    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # HMI histogram
    ax = axes[0]
    ax.hist(hm_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(hm_values), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(hm_values):.3f}')
    ax.axvline(np.median(hm_values), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(hm_values):.3f}')
    ax.set_xlabel('Houtman-Maks Index (Outlier Fraction)')
    ax.set_ylabel('Number of Households')
    ax.set_title(f'Robustness Distribution (n={len(hm_values)})\n(Lower = More Robust Rationality)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # HMI vs AEI scatter
    ax = axes[1]
    ax.scatter(aei_values, hm_values, alpha=0.5, s=25, c='coral')

    # Add correlation
    corr, p_val = stats.pearsonr(aei_values, hm_values)
    ax.set_xlabel('Integrity Score (AEI)')
    ax.set_ylabel('Houtman-Maks Index (Outlier Fraction)')
    ax.set_title(f'Efficiency vs Robustness\n(Pearson r={corr:.3f}, p={p_val:.2e})')
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(aei_values, hm_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(aei_values.min(), aei_values.max(), 100)
    ax.plot(x_line, p(x_line), 'b-', linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'showcase_j_houtman_maks.png', dpi=150)
    plt.close()

    # Find outlier-concentrated households (few outliers explain much)
    df = pd.DataFrame(household_data)
    df['outlier_concentration'] = df['n_removed'] / df['n_total']

    # Households where removing <10% of observations makes them consistent
    easy_fixes = df[df['hm_fraction'] < 0.1].sort_values('hm_fraction')

    return {
        'n_samples': len(hm_values),
        'mean_hm': round(float(np.mean(hm_values)), 4),
        'median_hm': round(float(np.median(hm_values)), 4),
        'std_hm': round(float(np.std(hm_values)), 4),
        'pct_perfect': round(float(np.mean(hm_values == 0)) * 100, 1),
        'pct_easy_fix': round(float(np.mean(hm_values < 0.1)) * 100, 1),
        'aei_hm_correlation': round(float(corr), 3),
        'easy_fix_examples': easy_fixes.head(5).to_dict('records') if len(easy_fixes) > 0 else [],
    }


def run_encoder_analysis() -> dict:
    """Run all encoder-based analyses."""
    print("=" * 70)
    print(" ENCODER-BASED PYREVEALED ANALYSIS")
    print("=" * 70)

    sessions = load_sessions()
    print(f"  Loaded {len(sessions)} sessions")

    results = {}

    # Showcase H
    results['preference_features'] = analyze_preference_features(sessions)

    # Showcase I
    results['auto_groups'] = analyze_auto_groups(sessions)

    # Showcase J
    results['houtman_maks'] = analyze_houtman_maks(sessions)

    print("\n" + "=" * 70)
    print(" ENCODER ANALYSIS COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_encoder_analysis()

    print("\n" + "=" * 70)
    print(" KEY FINDINGS")
    print("=" * 70)

    if 'error' not in results.get('preference_features', {}):
        print("\n[H] PREFERENCE ENCODER FEATURES")
        pf = results['preference_features']
        print(f"  Households fitted: {pf['n_fitted']}")
        print(f"  Clusters: {pf['n_clusters']}")
        for cs in pf['cluster_summary']:
            print(f"    Cluster {cs['cluster']}: n={cs['n_households']}, "
                  f"latent={cs['mean_latent']:.4f}, marginal={cs['mean_marginal']:.4f}, "
                  f"AEI={cs['mean_aei']:.3f}")

    if 'error' not in results.get('auto_groups', {}):
        print("\n[I] AUTO-DISCOVERED PRODUCT GROUPS")
        ag = results['auto_groups']
        print(f"  Households analyzed: {ag['n_households']}")
        print(f"  Confirmed manual groups: {ag['confirmed_manual_groups']}")
        print(f"  Manual group agreement:")
        for g, s in ag['manual_agreement'].items():
            status = "confirmed" if s > 0.5 else "not confirmed"
            print(f"    {g}: {s:.2f} ({status})")

    if 'error' not in results.get('houtman_maks', {}):
        print("\n[J] HOUTMAN-MAKS ROBUSTNESS")
        hm = results['houtman_maks']
        print(f"  Households tested: {hm['n_samples']}")
        print(f"  Mean outlier fraction: {hm['mean_hm']:.3f}")
        print(f"  Perfect rationality (HM=0): {hm['pct_perfect']:.1f}%")
        print(f"  Easy fixes (HM<0.1): {hm['pct_easy_fix']:.1f}%")
        print(f"  AEI-HM correlation: {hm['aei_hm_correlation']:.3f}")

    print(f"\nVisualizations saved to: {OUTPUT_DIR}")
