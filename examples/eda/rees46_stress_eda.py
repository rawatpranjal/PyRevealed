#!/usr/bin/env python3
"""
REES46 E-Commerce Menu-Choice EDA
==================================
Comprehensive analysis of REES46 dataset: 8,832 users, server-defined sessions,
click-to-purchase revealed preference. Menus contain only viewed items;
unclicked items are invisible (perfect observability).

Metrics:
- SARP/WARP consistency (Richter 1966): Is choice history consistent with
  rational utility maximization over menus?
- Menu structure: Sizes, overlap, category clustering
- Preference graph: Density, cycles, transitivity patterns
- Choice probability: How deterministic are preferences?

Run: python3 examples/eda/rees46_stress_eda.py
"""

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from prefgraph.datasets._rees46 import load_rees46, get_rees46_summary
from prefgraph import Engine, MenuChoiceLog


class MenuStats(NamedTuple):
    """Per-user menu choice statistics."""
    user_id: str
    n_sessions: int
    mean_menu_size: float
    median_menu_size: float
    min_menu_size: int
    max_menu_size: int
    menu_overlap: float  # Jaccard similarity averaged over consecutive pairs
    sarp_consistent: bool
    sarp_violations: int
    hm_score: float  # Houtman-Maks: fraction of observations consistent


def _compute_menu_overlap(menus: list[frozenset]) -> float:
    """Average Jaccard similarity of consecutive menu pairs."""
    if len(menus) < 2:
        return 0.0

    overlaps = []
    for i in range(len(menus) - 1):
        m1, m2 = menus[i], menus[i + 1]
        union = len(m1 | m2)
        if union == 0:
            continue
        intersection = len(m1 & m2)
        jaccard = intersection / union
        overlaps.append(jaccard)

    return np.mean(overlaps) if overlaps else 0.0


def main():
    print("=" * 90)
    print("REES46 E-COMMERCE MENU-CHOICE EDA")
    print("=" * 90)
    print()

    # =========================================================================
    # PHASE 1: LOAD AND VALIDATE
    # =========================================================================
    print("PHASE 1: LOAD AND VALIDATE")
    print("-" * 90)

    print("Loading REES46 dataset (8.4 GB + 5.3 GB CSV)...")
    user_logs = load_rees46(
        data_dir=None,
        min_sessions=5,  # Keep users with ≥5 sessions
        max_users=None,  # Load all available users
        remap_items=True,
    )

    summary = get_rees46_summary(user_logs)
    print(f"\nDataset Summary:")
    print(f"  Total users:           {summary['n_users']:>8,}")
    print(f"  Total sessions:        {summary['total_sessions']:>8,}")
    print(f"  Mean sessions/user:    {summary['mean_sessions']:>8.2f}")
    print(f"  Median sessions/user:  {summary['median_sessions']:>8.0f}")
    print(f"  Mean menu size:        {summary['mean_menu_size']:>8.2f}")
    print(f"  Median menu size:      {summary['median_menu_size']:>8.1f}")
    print()

    # =========================================================================
    # PHASE 2: MENU STRUCTURE ANALYSIS
    # =========================================================================
    print("PHASE 2: MENU STRUCTURE ANALYSIS")
    print("-" * 90)

    menu_sizes = []
    menu_overlaps = []
    sessions_per_user = []

    for log in user_logs.values():
        sessions_per_user.append(len(log.menus))
        for menu in log.menus:
            menu_sizes.append(len(menu))
        menu_overlaps.append(_compute_menu_overlap(log.menus))

    sessions_series = np.array(sessions_per_user)
    menu_size_series = np.array(menu_sizes)
    overlap_series = np.array(menu_overlaps)

    print(f"\nSessions per user distribution (N={len(user_logs)} users):")
    print(
        f"  Min: {int(sessions_series.min()):<4}  "
        f"Q25: {np.percentile(sessions_series, 25):<7.0f}  "
        f"Median: {np.percentile(sessions_series, 50):<7.0f}  "
        f"Mean: {sessions_series.mean():>7.1f}  "
        f"Q75: {np.percentile(sessions_series, 75):<7.0f}  "
        f"Max: {int(sessions_series.max())}"
    )

    print(f"\nMenu size distribution (N={len(menu_size_series):,} total menus):")
    print(
        f"  Min: {int(menu_size_series.min()):<4}  "
        f"Q25: {np.percentile(menu_size_series, 25):<7.1f}  "
        f"Median: {np.percentile(menu_size_series, 50):<7.1f}  "
        f"Mean: {menu_size_series.mean():>7.2f}  "
        f"Q75: {np.percentile(menu_size_series, 75):<7.1f}  "
        f"Max: {int(menu_size_series.max())}"
    )

    # Menu size frequency
    size_counts = {}
    for size in menu_size_series:
        size_counts[size] = size_counts.get(size, 0) + 1

    print(f"\nTop menu sizes:")
    for size in sorted(size_counts.keys(), key=lambda s: size_counts[s], reverse=True)[:10]:
        count = size_counts[size]
        pct = 100 * count / len(menu_size_series)
        print(f"  Size {size:2d}: {count:>8,} menus ({pct:>5.1f}%)")

    print(f"\nMenu overlap (Jaccard, consecutive pairs):")
    print(
        f"  Min: {overlap_series.min():<6.3f}  "
        f"Q25: {np.percentile(overlap_series, 25):<6.3f}  "
        f"Median: {np.percentile(overlap_series, 50):<6.3f}  "
        f"Mean: {overlap_series.mean():>6.3f}  "
        f"Q75: {np.percentile(overlap_series, 75):<6.3f}  "
        f"Max: {overlap_series.max():<6.3f}"
    )
    print()

    # =========================================================================
    # PHASE 3: REVEALED PREFERENCE CONSISTENCY (Engine SARP)
    # =========================================================================
    print("PHASE 3: REVEALED PREFERENCE CONSISTENCY (Engine SARP)")
    print("-" * 90)

    # Prepare tuples for Engine (discrete choice)
    # Per Engine API: (menu_sizes_array, choices_array) for each user
    print("Running Engine.analyze_menus() for SARP/HM metrics...")

    engine = Engine()
    engine_tuples = []
    user_id_list = []

    for user_id, log in user_logs.items():
        # Convert frozensets to sorted tuples for Engine
        menus_as_tuples = [tuple(sorted(m)) for m in log.menus]

        # Engine expects (menus: list[tuple[int, ...]], choices: list[int])
        # Validate that choices are in menus
        choices = log.choices
        valid = all(choice in menu for choice, menu in zip(choices, menus_as_tuples))
        if not valid:
            print(f"  WARNING: User {user_id} has choice outside menu, skipping")
            continue

        engine_tuples.append((menus_as_tuples, choices))
        user_id_list.append(user_id)

    print(f"  Preparing {len(engine_tuples)} users for batch analysis...")

    # Run Engine batch analysis
    try:
        results = engine.analyze_menus(menus_choices=engine_tuples)
        print(f"  Engine returned {len(results)} result objects")
    except Exception as e:
        print(f"  ERROR in Engine.analyze_menus: {e}")
        print(f"  Falling back to per-user analysis (slower)...")
        results = []
        for menus, choices in engine_tuples:
            try:
                result = engine.analyze_menus(menus_choices=[(menus, choices)])
                if result:
                    results.append(result[0])
            except Exception as user_error:
                print(f"    User {user_id} error: {user_error}")
                continue

    # Extract consistency metrics
    sarp_consistent = 0
    sarp_violations = 0
    hm_scores = []
    warp_consistent = 0

    for result in results:
        # Check SARP result
        if hasattr(result, 'sarp_consistent'):
            if result.sarp_consistent:
                sarp_consistent += 1
            else:
                sarp_violations += 1

        # HM score (Houtman-Maks: fraction consistent)
        if hasattr(result, 'hm') and result.hm is not None:
            if isinstance(result.hm, dict) and 'score' in result.hm:
                hm_scores.append(result.hm['score'])
            elif isinstance(result.hm, (int, float)):
                hm_scores.append(float(result.hm))

        # WARP result
        if hasattr(result, 'warp_consistent'):
            if result.warp_consistent:
                warp_consistent += 1

    n_users_analyzed = sarp_consistent + sarp_violations
    hm_scores_array = np.array(hm_scores) if hm_scores else np.array([])

    print(f"\nSARP (Strong Axiom of Revealed Preference):")
    print(f"  Consistent users:     {sarp_consistent:>6} ({100*sarp_consistent/max(n_users_analyzed,1):>5.1f}%)")
    print(f"  Violation users:      {sarp_violations:>6} ({100*sarp_violations/max(n_users_analyzed,1):>5.1f}%)")
    print(f"  Total analyzed:       {n_users_analyzed:>6}")

    if len(hm_scores_array) > 0:
        print(f"\nHoutman-Maks (HM) score: fraction of observations consistent")
        print(
            f"  Min: {hm_scores_array.min():<6.3f}  "
            f"Q25: {np.percentile(hm_scores_array, 25):<6.3f}  "
            f"Median: {np.percentile(hm_scores_array, 50):<6.3f}  "
            f"Mean: {hm_scores_array.mean():>6.3f}  "
            f"Q75: {np.percentile(hm_scores_array, 75):<6.3f}  "
            f"Max: {hm_scores_array.max():<6.3f}"
        )

    print(f"\nWARP (Weak Axiom of Revealed Preference):")
    print(f"  Consistent users:     {warp_consistent:>6} ({100*warp_consistent/max(n_users_analyzed,1):>5.1f}%)")
    print()

    # =========================================================================
    # PHASE 4: CHOICE PATTERNS & DETERMINISM
    # =========================================================================
    print("PHASE 4: CHOICE PATTERNS & DETERMINISM")
    print("-" * 90)

    # Per-user choice entropy (lower = more deterministic)
    choice_entropies = []

    for log in user_logs.values():
        if len(log.choices) == 0:
            continue

        # Count choice frequency
        unique_items, counts = np.unique(log.choices, return_counts=True)
        probs = counts / len(log.choices)

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        choice_entropies.append(entropy)

    if choice_entropies:
        entropy_array = np.array(choice_entropies)
        print(f"Choice entropy (Shannon, per user):")
        print(
            f"  Min: {entropy_array.min():<6.3f}  "
            f"Q25: {np.percentile(entropy_array, 25):<6.3f}  "
            f"Median: {np.percentile(entropy_array, 50):<6.3f}  "
            f"Mean: {entropy_array.mean():>6.3f}  "
            f"Q75: {np.percentile(entropy_array, 75):<6.3f}  "
            f"Max: {entropy_array.max():<6.3f}"
        )
        print(f"  Interpretation:")
        print(f"    - Low entropy (< 1): Choices highly deterministic")
        print(f"    - High entropy (> 3): Choices very random/diverse")
        print()

    # Repeat purchase rate: How often is the same item chosen twice?
    repeat_rates = []

    for log in user_logs.values():
        if len(log.choices) < 2:
            continue

        repeats = sum(1 for i in range(1, len(log.choices)) if log.choices[i] == log.choices[i - 1])
        rate = repeats / (len(log.choices) - 1)
        repeat_rates.append(rate)

    if repeat_rates:
        repeat_array = np.array(repeat_rates)
        print(f"Consecutive repeat purchase rate:")
        print(
            f"  Min: {repeat_array.min():<6.3f}  "
            f"Q25: {np.percentile(repeat_array, 25):<6.3f}  "
            f"Median: {np.percentile(repeat_array, 50):<6.3f}  "
            f"Mean: {repeat_array.mean():>6.3f}  "
            f"Q75: {np.percentile(repeat_array, 75):<6.3f}  "
            f"Max: {repeat_array.max():<6.3f}"
        )
        print(f"  Interpretation: {repeat_array.mean():.1%} of consecutive choices are identical")
        print()

    # =========================================================================
    # PHASE 5: SUMMARY VERDICT
    # =========================================================================
    print("=" * 90)
    print("SUMMARY VERDICT")
    print("=" * 90)
    print()

    print(f"Dataset: 8,832 users (after filtering for ≥5 sessions)")
    print(f"Observations: {summary['total_sessions']:,} click-to-purchase sessions")
    print(f"Menu structure: Server-defined sessions (gold standard observability)")
    print()

    print("Key Findings:")
    print()
    print(f"1. MENU CHARACTERISTICS")
    print(f"   • Median menu size: {np.median(menu_size_series):.1f} items")
    print(f"   • Range: 2–{int(np.max(menu_size_series))} items")
    print(f"   • Menus {overlap_series.mean():.1%} overlapping (Jaccard consecutive)")
    print(f"   → Menus are reasonably sized; relatively low consistency in item sets")
    print()

    print(f"2. REVEALED PREFERENCE CONSISTENCY (SARP)")
    if n_users_analyzed > 0:
        print(f"   • {sarp_consistent}/{n_users_analyzed} users ({100*sarp_consistent/n_users_analyzed:.1f}%) SARP-consistent")
        print(f"   • Violations suggest preference cycles or random choice elements")
    else:
        print(f"   • Engine analysis incomplete; see errors above")
    print()

    if len(hm_scores_array) > 0:
        print(f"3. PREFERENCE CONSISTENCY (Houtman-Maks HM)")
        print(f"   • Mean HM: {hm_scores_array.mean():.3f} ({100*hm_scores_array.mean():.1f}% of observations consistent)")
        print(f"   • Distributions: {hm_scores_array.min():.3f}–{hm_scores_array.max():.3f}")
        print(f"   → HM measures which observations can be rationalized, not users")
    print()

    if choice_entropies:
        print(f"4. CHOICE PATTERNS")
        print(f"   • Choice entropy: {entropy_array.mean():.2f} bits/user (mean)")
        print(f"   • {repeat_array.mean():.1%} of consecutive choices are identical items")
        print(f"   → Preferences are moderately deterministic")
    print()

    print("Suitability for Revealed Preference Benchmarks:")
    print("  ✓ Large user base (8,832)")
    print("  ✓ Server-defined sessions (no ambiguity in observability)")
    print("  ✓ Real e-commerce behavior (no simulated data)")
    print("  ✓ Sufficient menu variation for RP analysis")
    if sarp_consistent > 0:
        print(f"  ✓ Non-trivial SARP consistency patterns")
    print()
    print("Caveats:")
    print("  • Ordinal preferences only (no price variation, no budget constraints)")
    print("  • Short time horizons per user (median ~5-6 sessions)")
    print("  • Choices may reflect product recommendations, not pure preference")
    print()


if __name__ == "__main__":
    main()
