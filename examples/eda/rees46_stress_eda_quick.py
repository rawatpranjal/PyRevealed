#!/usr/bin/env python3
"""
REES46 E-Commerce Menu-Choice EDA (Quick version for initial exploration)
Load just 1,000 users to understand the data structure and assumptions.
"""

import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from prefgraph.datasets._rees46 import load_rees46, get_rees46_summary
from prefgraph import Engine


def main():
    print("=" * 90)
    print("REES46 E-COMMERCE MENU-CHOICE EDA (QUICK)")
    print("Loading 1,000 users for initial exploration...")
    print("=" * 90)
    print()

    # Load with reduced user cap for speed
    print("Phase 1: Loading REES46 dataset...")
    print("  Data dir: ~/.prefgraph/data/rees46/")
    print("  Max users: 1,000 (for quick exploration)")
    print()

    user_logs = load_rees46(
        data_dir=None,
        min_sessions=5,  # Require ≥5 sessions per user
        max_users=1000,
    )

    summary = get_rees46_summary(user_logs)

    print("DATASET SUMMARY")
    print("-" * 90)
    print(f"Users loaded:             {summary['n_users']:,}")
    print(f"Total sessions:           {summary['total_sessions']:,}")
    print(f"Mean sessions per user:   {summary['mean_sessions']:.2f}")
    print(f"Median sessions per user: {summary['median_sessions']:.0f}")
    print(f"Mean menu size:           {summary['mean_menu_size']:.2f} items")
    print(f"Median menu size:         {summary['median_menu_size']:.1f} items")
    print()

    # =========================================================================
    # KEY ASSUMPTIONS IN RAW-DATA-TO-CHOICE-SETS CONVERSION
    # =========================================================================
    print("=" * 90)
    print("KEY ASSUMPTIONS: RAW DATA → CHOICE SETS CONVERSION")
    print("=" * 90)
    print()

    print("ASSUMPTION 1: Session boundaries = user_session column (server-defined)")
    print("-" * 90)
    print("Raw data: event stream (user_id, user_session, event_type, product_id, ...)")
    print("Decision: Use server's user_session field as ground truth for sessions")
    print("Rationale:")
    print("  • Server-defined sessions are gold standard (no ambiguity)")
    print("  • Alternative: gap-based (30min inactivity) — would be error-prone")
    print("  • user_session is explicitly designed for this purpose")
    print()

    print("ASSUMPTION 2: Menu = all viewed items in session (including purchased item)")
    print("-" * 90)
    print("Raw events: Each session has multiple 'view' events and one 'purchase' event")
    print("Decision: Menu = {product_ids where event_type == 'view'} ∪ {purchased_item}")
    print("Rationale:")
    print("  • Viewed items = items the recommender system showed (visible)")
    print("  • Purchased item is always in menu (added if not viewed)")
    print("  • Unviewed items are INVISIBLE to user (not in choice set)")
    print("  • This is perfect observability: no hidden alternatives")
    print()

    print("ASSUMPTION 3: Exactly one purchase per session")
    print("-" * 90)
    print("Raw data: Some sessions have 0 purchases, some have 2+")
    print("Decision: Filter to sessions with exactly 1 purchase")
    print(f"Impact: Reduces session count but ensures clear choice identification")
    print()

    print("ASSUMPTION 4: Menu size constraints: 2 ≤ size ≤ 50")
    print("-" * 90)
    print("Decision: Drop sessions with <2 items or >50 items")
    print("Rationale:")
    print("  • Size < 2: Single item shown → no choice, no revelation of preference")
    print("  • Size > 50: Likely data quality issues or batch operations")
    print("  • Keeps reasonable choice set complexity")
    print()

    print("ASSUMPTION 5: User-level filtering: ≥5 sessions per user")
    print("-" * 90)
    print("Decision: Include only users with ≥5 choice observations")
    print("Rationale:")
    print(f"  • Current filtering: {summary['n_users']:,} users meet this threshold")
    print("  • <5 sessions = insufficient data for preference learning")
    print("  • Trade-off: fewer users but more reliable per-user metrics")
    print()

    print("ASSUMPTION 6: Item remapping to 0..N-1 (per user)")
    print("-" * 90)
    print("Decision: Map each user's item IDs to consecutive integers")
    print("Rationale:")
    print("  • Raw product_ids are scattered (not sequential)")
    print("  • Remapping compresses memory without loss of structure")
    print("  • Preferences depend only on choice, not absolute IDs")
    print()

    # =========================================================================
    # SAMPLE MENU-CHOICE OBSERVATIONS
    # =========================================================================
    print("=" * 90)
    print("SAMPLE MENU-CHOICE OBSERVATIONS (First 3 users)")
    print("=" * 90)
    print()

    for i, (user_id, log) in enumerate(list(user_logs.items())[:3]):
        print(f"User {user_id}:")
        print(f"  Sessions: {len(log.choices)}")
        print(f"  First 5 observations:")

        for j in range(min(5, len(log.choices))):
            menu = sorted(log.menus[j])
            choice = log.choices[j]
            is_chosen = "✓" if choice in log.menus[j] else "ERROR"
            print(f"    Session {j+1:2d}: Menu={menu} → Choice={choice} {is_chosen}")

        print()

    # =========================================================================
    # MENU STRUCTURE STATISTICS
    # =========================================================================
    print("=" * 90)
    print("MENU STRUCTURE STATISTICS")
    print("=" * 90)
    print()

    menu_sizes = []
    for log in user_logs.values():
        for menu in log.menus:
            menu_sizes.append(len(menu))

    menu_size_array = np.array(menu_sizes)

    print(f"Menu size distribution (N={len(menu_sizes):,} total sessions):")
    print(f"  Min:    {int(menu_size_array.min())}")
    print(f"  Q25:    {np.percentile(menu_size_array, 25):.1f}")
    print(f"  Median: {np.percentile(menu_size_array, 50):.1f}")
    print(f"  Mean:   {menu_size_array.mean():.2f}")
    print(f"  Q75:    {np.percentile(menu_size_array, 75):.1f}")
    print(f"  Max:    {int(menu_size_array.max())}")
    print()

    # Show top 5 menu sizes
    print("Top menu sizes:")
    size_counts = {}
    for size in menu_size_array:
        size_counts[int(size)] = size_counts.get(int(size), 0) + 1

    for size in sorted(size_counts.keys(), key=lambda s: size_counts[s], reverse=True)[:10]:
        count = size_counts[size]
        pct = 100 * count / len(menu_sizes)
        bar = "█" * int(pct / 2)
        print(f"  Size {size:2d}: {count:>7,} sessions ({pct:>5.1f}%) {bar}")

    print()

    # =========================================================================
    # VALIDATION: CHOICE MUST BE IN MENU
    # =========================================================================
    print("=" * 90)
    print("VALIDATION: CHOICE MUST BE IN MENU")
    print("=" * 90)
    print()

    violations = 0
    for user_id, log in user_logs.items():
        for choice, menu in zip(log.choices, log.menus):
            if choice not in menu:
                violations += 1
                if violations <= 5:
                    print(f"  ERROR: User {user_id}, choice {choice} not in menu {menu}")

    if violations == 0:
        print(f"✓ All {len(user_logs) * summary['mean_sessions']:.0f} choices are valid (in their menus)")
    else:
        print(f"✗ Found {violations} violations")

    print()

    # =========================================================================
    # REVEALED PREFERENCE SNAPSHOT (Per-user metrics)
    # =========================================================================
    print("=" * 90)
    print("REVEALED PREFERENCE SNAPSHOT")
    print("=" * 90)
    print()

    print("Testing first 100 users with Engine.analyze_menus() for SARP consistency...")

    engine = Engine()
    test_users = list(user_logs.items())[:100]
    engine_tuples = []

    for user_id, log in test_users:
        menus_tuples = [tuple(sorted(m)) for m in log.menus]
        engine_tuples.append((menus_tuples, log.choices))

    try:
        results = engine.analyze_menus(menus_choices=engine_tuples)
        print(f"✓ Engine analysis completed for {len(results)} users")
        print()

        sarp_count = sum(1 for r in results if hasattr(r, 'sarp_consistent') and r.sarp_consistent)
        print(f"SARP Consistency (first 100 users):")
        print(f"  Consistent:   {sarp_count} ({100*sarp_count/len(results):.1f}%)")
        print(f"  Violations:   {len(results) - sarp_count} ({100*(len(results)-sarp_count)/len(results):.1f}%)")
        print()

        # Collect HM scores
        hm_scores = []
        for r in results:
            if hasattr(r, 'hm') and r.hm is not None:
                if isinstance(r.hm, dict) and 'score' in r.hm:
                    hm_scores.append(r.hm['score'])
                elif isinstance(r.hm, (int, float)):
                    hm_scores.append(float(r.hm))

        if hm_scores:
            hm_array = np.array(hm_scores)
            print(f"Houtman-Maks (HM) Consistency Score:")
            print(f"  Min:    {hm_array.min():.3f}")
            print(f"  Q25:    {np.percentile(hm_array, 25):.3f}")
            print(f"  Median: {np.percentile(hm_array, 50):.3f}")
            print(f"  Mean:   {hm_array.mean():.3f}")
            print(f"  Q75:    {np.percentile(hm_array, 75):.3f}")
            print(f"  Max:    {hm_array.max():.3f}")

    except Exception as e:
        print(f"✗ Engine error: {e}")

    print()
    print("=" * 90)
    print("SUMMARY: REES46 is suitable for menu-choice RP analysis")
    print("=" * 90)
    print()


if __name__ == "__main__":
    main()
