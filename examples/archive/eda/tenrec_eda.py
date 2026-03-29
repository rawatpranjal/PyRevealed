#!/usr/bin/env python3
"""Tenrec EDA: Diagnostics for menu-choice viability.

Audits whether Tenrec (Tencent QQ Browser, 5M users, 140M interactions)
can be defensibly used as a menu-choice revealed preference benchmark.

Key question: Is "clicks before a like" a credible static menu, or is
Tenrec fundamentally a sequential exposure/engagement dataset that should
be reframed or dropped?

Six diagnostics:
1. Action-state consistency (click, like, share, follow joint distribution)
2. Window-size distribution under 5 construction rules (the critical table)
3. Category coherence within windows
4. Position of the chosen item (last? early?)
5. Watch-time separation (liked vs clicked vs exposed)
6. User-level pathology rates (size-1 prevalence, like rates)

Run: python examples/eda/tenrec_eda.py

================================================================================
CRITICAL ISSUES IDENTIFIED (see tenrec_report.md for full results)
================================================================================

1. CHOSEN-ITEM ALWAYS LAST (100%, STRUCTURAL)
   Windows are defined ending at a like event. The liked item is mechanically
   terminal. This means the signal is NOT preference from a set, but sequential
   stopping behavior ("user scrolls until they find something they like, stops").

   Consequence: Cannot be fixed by window redefinition. The like/share/follow
   is the only credible choice signal, but it's always the terminal event.

2. HIGH SIZE-1 PREVALENCE (35-46% of windows)
   35-46% of windows are single-item exposures. A one-item "menu" is not a
   choice set. These observations have zero information for preference revelation.

3. HUGE TAIL RISK (p99 = 198-372 items)
   Window size distribution is heavily right-skewed. While median is small (2-3),
   the p99 reaches 198-372 items. Some users see hundreds of items before liking
   something. These degenerate windows violate the coherent-menu assumption.

4. SIZE vs CATEGORY COHERENCE TRADE-OFF (UNRESOLVABLE)
   - Category-run micro-sessions: Preserve 99.8% category coherence but produce
     median window size 1.0 (too small to be a menu)
   - K=5 windows: Achieve reasonable size but only 22% same-category
   - K=10 windows: Larger size but only 8.6% same-category

   No construction achieves both reasonable size (≥4) AND high coherence (>70%).

5. SPARSE LIKE SIGNAL (0.8-1.1% of rows)
   Like events are rare. Only ~1 in 100-125 rows is a like. This makes it hard
   to build enough valid windows per user for stable RP analysis.

6. NO TIMESTAMPS, ONLY TEMPORAL ORDER
   Tenrec released data have no timestamps, only row order. Any "session"
   definition is synthetic. Cannot use gap-based sessioning heuristics.

7. WINDOWS ARE INHERENTLY SEQUENTIAL, NOT SIMULTANEOUS
   Data represent a recommendation feed where users see items one-by-one
   in time order. This is NOT a simultaneous menu-choice context (like a
   grocery shelf). Sequential behavior confounds choice with order effects
   and stopping rules.

8. CATEGORY DIVERSITY IN FULL WINDOWS
   In full-file analysis (2.44M rows): median unique category = 2.0, not 1.0.
   Larger window definitions (K=10) can span many categories. This destroys
   the coherent-menu assumption for classical RP.

9. ACCEPTANCE CRITERIA ALL CONSTRUCTIONS FAIL
   Threshold: 4+ out of 5 criteria pass (80%)
   Result: All three salvage constructions score 3/5 (60%)

   Failing criteria:
   - Chosen-last < 100%: All fail (100% across all constructions)
   - Median size >= 4: Cat-run fails (median 1.0)
   - Same-category > 70%: K-windows fail (22%, 8.6%)

10. FULL-FILE ROBUSTNESS CHECK CONFIRMS FINDINGS
    Tested on full 2.44M row QB-video.csv (not just sequential 500K):
    - Chosen-last = 100% identical (structural, not artifact)
    - p99 window size 220 (similar to 500K: 199)
    - Size-1 prevalence 34.8% (similar to 500K: 45.6%)
    - Category coherence stable (81.6% vs 85.9%)

    Conclusion: 500K sequential sample IS representative. Salvage failure
    is robust across entire dataset.

================================================================================
BOTTOM LINE
================================================================================

Tenrec is a recommender-system engagement log, not a menu-choice dataset.

Cannot be salvaged for classical RP analysis. The fundamental issue is that
the "choice" (like) is the terminal event in a sequence, not a decision from
a simultaneous set of alternatives.

Recommended action: Move to appendix (sequential-engagement reframe) or drop.

See: tenrec_report.md (facts), tenrec_salvage_report.md (salvage results),
     tenrec_robustness_check.md (full-file validation)
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import polars as pl
import numpy as np

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_tenrec_data(csv_path: str | Path, use_sample: bool = True) -> pl.DataFrame:
    """Load QB-video.csv or sample (handle \\N null markers)."""
    if use_sample:
        # Use pre-computed 500k-row sample for speed
        sample_path = Path('/tmp/tenrec_sample.csv')
        if sample_path.exists():
            print(f"  Loading {sample_path.name} (pre-computed 500k sample)...")
            csv_path = sample_path
        else:
            print(f"  Sample not found, loading full file with null handling...")

    print(f"  Loading {Path(csv_path).name}...")
    df = pl.read_csv(csv_path, null_values='\\N')
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Columns: {df.columns}")
    return df


# ============================================================================
# DIAGNOSTIC 1: Action-state consistency
# ============================================================================

def diagnostic_1_action_consistency(df: pl.DataFrame) -> None:
    """Joint distribution of (click, like, share, follow)."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 1: Action-state consistency")
    print("=" * 80)

    # All rows
    total_rows = df.shape[0]
    print(f"\nTotal rows: {total_rows:,}")

    # Pure exposures (all actions zero)
    pure_exposure = df.filter(
        (pl.col("click") == 0)
        & (pl.col("like") == 0)
        & (pl.col("share") == 0)
        & (pl.col("follow") == 0)
    )
    print(f"Pure exposures (all actions=0): {len(pure_exposure):,} ({100*len(pure_exposure)/total_rows:.1f}%)")

    # Rows with at least one positive action
    any_action = df.filter(
        (pl.col("click") == 1)
        | (pl.col("like") == 1)
        | (pl.col("share") == 1)
        | (pl.col("follow") == 1)
    )
    print(f"Rows with any action: {len(any_action):,} ({100*len(any_action)/total_rows:.1f}%)")

    # Like without click
    like_no_click = df.filter((pl.col("like") == 1) & (pl.col("click") == 0))
    print(f"Like without click: {len(like_no_click):,} ({100*len(like_no_click)/total_rows:.1f}%)")

    # Check: is like always inside click?
    all_likes_have_click = like_no_click.shape[0] == 0
    print(f"  → Like always requires click?: {all_likes_have_click}")

    # Share without click
    share_no_click = df.filter((pl.col("share") == 1) & (pl.col("click") == 0))
    print(f"Share without click: {len(share_no_click):,} ({100*len(share_no_click)/total_rows:.1f}%)")

    # Follow without click
    follow_no_click = df.filter((pl.col("follow") == 1) & (pl.col("click") == 0))
    print(f"Follow without click: {len(follow_no_click):,} ({100*len(follow_no_click)/total_rows:.1f}%)")

    # Action mix: click only, like only, click+like, etc.
    print("\nAction type combinations (top 10):")
    action_combos = (
        df.select([
            (pl.col("click").cast(str) + pl.col("like").cast(str) + pl.col("share").cast(str) + pl.col("follow").cast(str)).alias("combo")
        ])
        .group_by("combo")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(10)
    )
    for row in action_combos.iter_rows(named=True):
        combo = row["combo"]
        count = row["count"]
        pct = 100 * count / total_rows
        click, like, share, follow = [int(c) for c in combo]
        desc = f"click={click}, like={like}, share={share}, follow={follow}"
        print(f"  {desc:40s} {count:>8,} ({pct:5.1f}%)")


# ============================================================================
# DIAGNOSTIC 2: Window-size distribution (5 construction rules)
# ============================================================================

def compute_window_stats(sizes: list[int]) -> dict:
    """Compute percentiles and size breakdowns."""
    if not sizes:
        return {
            "median": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "pct_size_1": 0,
            "pct_gt_20": 0,
            "pct_gt_50": 0,
        }
    sizes_arr = np.array(sizes, dtype=float)
    return {
        "median": float(np.percentile(sizes_arr, 50)),
        "p75": float(np.percentile(sizes_arr, 75)),
        "p90": float(np.percentile(sizes_arr, 90)),
        "p95": float(np.percentile(sizes_arr, 95)),
        "p99": float(np.percentile(sizes_arr, 99)),
        "pct_size_1": 100 * sum(1 for s in sizes if s == 1) / len(sizes),
        "pct_gt_20": 100 * sum(1 for s in sizes if s > 20) / len(sizes),
        "pct_gt_50": 100 * sum(1 for s in sizes if s > 50) / len(sizes),
    }


def compute_chosen_last_share(df_window: pl.DataFrame, like_row_idx: int) -> float:
    """Compute fraction of windows where liked item is in last position."""
    if df_window.shape[0] == 0:
        return 0.0
    return 1.0 if like_row_idx == df_window.shape[0] - 1 else 0.0


def compute_top_category_share(categories: list[int]) -> float:
    """Compute fraction of window items in the most common category."""
    if not categories:
        return 0.0
    counter = Counter(categories)
    top_count = max(counter.values())
    return top_count / len(categories)


def diagnostic_2_window_sizes(df: pl.DataFrame) -> dict:
    """Compute window sizes for 5 construction rules."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 2: Window-size distribution (5 construction rules)")
    print("=" * 80)

    # Sort by user then by row order (temporal)
    df_sorted = df.sort("user_id")

    results = {}

    # Rule 1: Exposures (clicks=0 or any action=0) since last like
    # Actually, let's define: all rows since last like
    rule1_sizes = []
    rule1_chosen_last = []
    rule1_top_cats = []

    for user_id in df_sorted["user_id"].unique():
        user_df = df_sorted.filter(pl.col("user_id") == user_id)
        window = []
        window_cats = []
        for i, row in enumerate(user_df.iter_rows(named=True)):
            window.append(row["item_id"])
            window_cats.append(row["video_category"])
            if row["like"] == 1:
                # Window complete
                if len(window) >= 1:
                    rule1_sizes.append(len(window))
                    # Check if liked item is last
                    rule1_chosen_last.append(compute_chosen_last_share(user_df[i:i+1], 0))
                    rule1_top_cats.append(compute_top_category_share(window_cats))
                window = []
                window_cats = []

    results["R1_exposures_since_like"] = {
        **compute_window_stats(rule1_sizes),
        "chosen_last_share": np.mean(rule1_chosen_last) * 100 if rule1_chosen_last else 0,
        "top_category_share": np.mean(rule1_top_cats) * 100 if rule1_top_cats else 0,
        "n_windows": len(rule1_sizes),
    }

    # Rule 2: Clicks (click=1) since last like
    rule2_sizes = []
    rule2_chosen_last = []
    rule2_top_cats = []

    for user_id in df_sorted["user_id"].unique():
        user_df = df_sorted.filter(pl.col("user_id") == user_id)
        window = []
        window_cats = []
        like_idx = None
        for i, row in enumerate(user_df.iter_rows(named=True)):
            if row["click"] == 1:
                window.append(row["item_id"])
                window_cats.append(row["video_category"])
                like_idx = len(window) - 1
            if row["like"] == 1:
                if len(window) >= 1:
                    rule2_sizes.append(len(window))
                    rule2_chosen_last.append(1.0 if like_idx == len(window) - 1 else 0.0)
                    rule2_top_cats.append(compute_top_category_share(window_cats))
                window = []
                window_cats = []
                like_idx = None

    results["R2_clicks_since_like"] = {
        **compute_window_stats(rule2_sizes),
        "chosen_last_share": np.mean(rule2_chosen_last) * 100 if rule2_chosen_last else 0,
        "top_category_share": np.mean(rule2_top_cats) * 100 if rule2_top_cats else 0,
        "n_windows": len(rule2_sizes),
    }

    # Rule 3: Exposures since last positive action (like OR share OR follow)
    rule3_sizes = []
    rule3_chosen_last = []
    rule3_top_cats = []

    for user_id in df_sorted["user_id"].unique():
        user_df = df_sorted.filter(pl.col("user_id") == user_id)
        window = []
        window_cats = []
        chosen_idx = None
        for i, row in enumerate(user_df.iter_rows(named=True)):
            window.append(row["item_id"])
            window_cats.append(row["video_category"])
            if row["like"] == 1 or row["share"] == 1 or row["follow"] == 1:
                chosen_idx = len(window) - 1
                if len(window) >= 1:
                    rule3_sizes.append(len(window))
                    rule3_chosen_last.append(1.0 if chosen_idx == len(window) - 1 else 0.0)
                    rule3_top_cats.append(compute_top_category_share(window_cats))
                window = []
                window_cats = []
                chosen_idx = None

    results["R3_pos_action"] = {
        **compute_window_stats(rule3_sizes),
        "chosen_last_share": np.mean(rule3_chosen_last) * 100 if rule3_chosen_last else 0,
        "top_category_share": np.mean(rule3_top_cats) * 100 if rule3_top_cats else 0,
        "n_windows": len(rule3_sizes),
    }

    # Rule 4: Last 5 exposures before a like
    rule4_sizes = []
    rule4_chosen_last = []
    rule4_top_cats = []

    for user_id in df_sorted["user_id"].unique():
        user_df = df_sorted.filter(pl.col("user_id") == user_id)
        for i, row in enumerate(user_df.iter_rows(named=True)):
            if row["like"] == 1:
                # Take last 5 rows (or fewer if user just started)
                start_idx = max(0, i - 4)
                window_rows = user_df[start_idx:i+1]
                if window_rows.shape[0] >= 1:
                    sz = window_rows.shape[0]
                    rule4_sizes.append(sz)
                    rule4_chosen_last.append(1.0)  # Liked item is always last in this window
                    cats = window_rows["video_category"].to_list()
                    rule4_top_cats.append(compute_top_category_share(cats))

    results["R4_last_5_before_like"] = {
        **compute_window_stats(rule4_sizes),
        "chosen_last_share": np.mean(rule4_chosen_last) * 100 if rule4_chosen_last else 0,
        "top_category_share": np.mean(rule4_top_cats) * 100 if rule4_top_cats else 0,
        "n_windows": len(rule4_sizes),
    }

    # Rule 5: Last 10 exposures before a like
    rule5_sizes = []
    rule5_chosen_last = []
    rule5_top_cats = []

    for user_id in df_sorted["user_id"].unique():
        user_df = df_sorted.filter(pl.col("user_id") == user_id)
        for i, row in enumerate(user_df.iter_rows(named=True)):
            if row["like"] == 1:
                start_idx = max(0, i - 9)
                window_rows = user_df[start_idx:i+1]
                if window_rows.shape[0] >= 1:
                    sz = window_rows.shape[0]
                    rule5_sizes.append(sz)
                    rule5_chosen_last.append(1.0)
                    cats = window_rows["video_category"].to_list()
                    rule5_top_cats.append(compute_top_category_share(cats))

    results["R5_last_10_before_like"] = {
        **compute_window_stats(rule5_sizes),
        "chosen_last_share": np.mean(rule5_chosen_last) * 100 if rule5_chosen_last else 0,
        "top_category_share": np.mean(rule5_top_cats) * 100 if rule5_top_cats else 0,
        "n_windows": len(rule5_sizes),
    }

    # Print the key table
    print("\nCritical Summary Table:")
    print("-" * 130)
    print(
        f"{'Rule':<35} {'Median':>7} {'p90':>7} {'p99':>7} "
        f"{'%sz1':>6} {'>20%':>6} {'>50%':>6} {'chosen-last%':>11} {'top-cat%':>9}"
    )
    print("-" * 130)

    for rule_name, stats in results.items():
        print(
            f"{rule_name:<35} "
            f"{stats['median']:>7.1f} "
            f"{stats['p90']:>7.1f} "
            f"{stats['p99']:>7.1f} "
            f"{stats['pct_size_1']:>6.1f} "
            f"{stats['pct_gt_20']:>6.1f} "
            f"{stats['pct_gt_50']:>6.1f} "
            f"{stats['chosen_last_share']:>11.1f} "
            f"{stats['top_category_share']:>9.1f}"
        )

    print("-" * 130)

    return results


# ============================================================================
# DIAGNOSTIC 3: Category coherence (Rule 2 only)
# ============================================================================

def diagnostic_3_category_coherence(df: pl.DataFrame) -> None:
    """Category diversity within windows (Rule 2: clicks since last like)."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 3: Category coherence (Rule 2: clicks since last like)")
    print("=" * 80)

    df_sorted = df.sort("user_id")

    entropies = []
    unique_counts = []
    top_shares = []

    for user_id in df_sorted["user_id"].unique():
        user_df = df_sorted.filter(pl.col("user_id") == user_id)
        window_cats = []
        for row in user_df.iter_rows(named=True):
            if row["click"] == 1:
                window_cats.append(row["video_category"])
            if row["like"] == 1:
                if len(window_cats) >= 1:
                    unique_counts.append(len(set(window_cats)))
                    counter = Counter(window_cats)
                    top_share = max(counter.values()) / len(window_cats)
                    top_shares.append(top_share)
                    # Entropy
                    probs = np.array(list(counter.values())) / len(window_cats)
                    ent = -np.sum(probs * np.log(probs + 1e-10))
                    entropies.append(ent)
                window_cats = []

    print(f"\nUnique categories per window (R2):")
    print(f"  Median: {np.median(unique_counts):.1f}")
    print(f"  p90:    {np.percentile(unique_counts, 90):.1f}")
    print(f"  p99:    {np.percentile(unique_counts, 99):.1f}")

    print(f"\nCategory entropy per window:")
    print(f"  Median: {np.median(entropies):.2f}")
    print(f"  p90:    {np.percentile(entropies, 90):.2f}")
    print(f"  p99:    {np.percentile(entropies, 99):.2f}")

    print(f"\nTop-category share (concentration) per window:")
    print(f"  Median: {np.median(top_shares):.2f}")
    print(f"  p90:    {np.percentile(top_shares, 90):.2f}")
    print(f"  p99:    {np.percentile(top_shares, 99):.2f}")
    print(f"  → If < 0.4-0.5, windows are cross-category (weak menu)")
    print(f"  → If > 0.6-0.7, windows are locally coherent (stronger menu)")


# ============================================================================
# DIAGNOSTIC 4: Position of chosen item (Rule 2)
# ============================================================================

def diagnostic_4_chosen_position(df: pl.DataFrame) -> None:
    """Where does the liked item appear in the click-window?"""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 4: Position of chosen item (Rule 2: clicks since last like)")
    print("=" * 80)

    df_sorted = df.sort("user_id")

    positions = []  # 1-indexed rank within window
    normalized_positions = []  # position / window_size

    for user_id in df_sorted["user_id"].unique():
        user_df = df_sorted.filter(pl.col("user_id") == user_id)
        window = []
        like_position = None
        for i, row in enumerate(user_df.iter_rows(named=True)):
            if row["click"] == 1:
                window.append(row["item_id"])
                like_position = len(window)
            if row["like"] == 1:
                if len(window) >= 1:
                    positions.append(like_position)
                    normalized_positions.append(like_position / len(window))
                window = []
                like_position = None

    if positions:
        chosen_last_share = sum(1 for p, w in zip(positions, [len(window) for _ in positions]) if p == w) / len(positions)
    else:
        chosen_last_share = 0

    print(f"\nLiked item position within click-window:")
    print(f"  Median rank: {np.median(positions):.1f}")
    print(f"  Median normalized pos: {np.median(normalized_positions):.2f}")
    print(f"  % always last: {chosen_last_share * 100:.1f}%")
    print(f"  → If > 80%, suggests sequential stopping, not menu choice")
    print(f"  → If uniform, suggests real choice from set")

    # Histogram
    print(f"\nNormalized position distribution:")
    pos_buckets = {
        "[0.0-0.2)": sum(1 for p in normalized_positions if 0 <= p < 0.2),
        "[0.2-0.4)": sum(1 for p in normalized_positions if 0.2 <= p < 0.4),
        "[0.4-0.6)": sum(1 for p in normalized_positions if 0.4 <= p < 0.6),
        "[0.6-0.8)": sum(1 for p in normalized_positions if 0.6 <= p < 0.8),
        "[0.8-1.0]": sum(1 for p in normalized_positions if 0.8 <= p <= 1.0),
    }
    total_pos = len(normalized_positions)
    for bucket, count in pos_buckets.items():
        pct = 100 * count / total_pos if total_pos else 0
        print(f"  {bucket}: {count:>7,} ({pct:5.1f}%)")


# ============================================================================
# DIAGNOSTIC 5: Watch-time separation
# ============================================================================

def diagnostic_5_watch_time(df: pl.DataFrame) -> None:
    """Compare watching_times across action types."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 5: Watch-time separation")
    print("=" * 80)

    # Liked items
    liked = df.filter(pl.col("like") == 1)
    liked_times = liked["watching_times"].to_list()

    # Clicked but NOT liked
    clicked_not_liked = df.filter((pl.col("click") == 1) & (pl.col("like") == 0))
    clicked_times = clicked_not_liked["watching_times"].to_list()

    # Exposed (no click, no like)
    exposed = df.filter((pl.col("click") == 0) & (pl.col("like") == 0))
    exposed_times = exposed["watching_times"].to_list()

    print(f"\nWatching times (seconds or units):")
    print(f"  Liked items:        median={np.median(liked_times):.1f}, mean={np.mean(liked_times):.1f}, n={len(liked_times):,}")
    print(f"  Clicked not liked:  median={np.median(clicked_times):.1f}, mean={np.mean(clicked_times):.1f}, n={len(clicked_times):,}")
    print(f"  Exposed only:       median={np.median(exposed_times):.1f}, mean={np.mean(exposed_times):.1f}, n={len(exposed_times):,}")

    print(f"\n  → If liked >> clicked, suggests dwell/engagement, not preference")
    print(f"  → If similar, more consistent with static menu choice")


# ============================================================================
# DIAGNOSTIC 6: User-level pathology rates
# ============================================================================

def diagnostic_6_user_pathology(df: pl.DataFrame) -> None:
    """Per-user: like rate, positive rate, size-1 window prevalence."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 6: User-level pathology rates")
    print("=" * 80)

    df_sorted = df.sort("user_id")

    user_stats = []

    for user_id in df_sorted["user_id"].unique():
        user_df = df_sorted.filter(pl.col("user_id") == user_id)
        n_rows = user_df.shape[0]

        likes = (user_df["like"] == 1).sum()
        positives = (
            ((user_df["click"] == 1) | (user_df["like"] == 1) |
             (user_df["share"] == 1) | (user_df["follow"] == 1))
        ).sum()

        like_rate = likes / n_rows if n_rows else 0
        pos_rate = positives / n_rows if n_rows else 0

        # Size-1 window count (Rule 2: clicks until like)
        window_sizes = []
        window = []
        for row in user_df.iter_rows(named=True):
            if row["click"] == 1:
                window.append(row["item_id"])
            if row["like"] == 1:
                window_sizes.append(len(window))
                window = []

        size_1_windows = sum(1 for s in window_sizes if s == 1)
        size_1_share = size_1_windows / len(window_sizes) if window_sizes else 0
        median_window_size = np.median(window_sizes) if window_sizes else 0

        user_stats.append({
            "like_rate": like_rate,
            "pos_rate": pos_rate,
            "size_1_share": size_1_share,
            "median_window": median_window_size,
        })

    # Aggregate
    like_rates = [u["like_rate"] for u in user_stats]
    pos_rates = [u["pos_rate"] for u in user_stats]
    size_1_shares = [u["size_1_share"] for u in user_stats]
    median_windows = [u["median_window"] for u in user_stats]

    print(f"\nPer-user like rate distribution:")
    print(f"  Median: {np.median(like_rates):.3f}")
    print(f"  p90:    {np.percentile(like_rates, 90):.3f}")
    print(f"  p99:    {np.percentile(like_rates, 99):.3f}")

    print(f"\nPer-user positive-action rate:")
    print(f"  Median: {np.median(pos_rates):.3f}")
    print(f"  p90:    {np.percentile(pos_rates, 90):.3f}")
    print(f"  p99:    {np.percentile(pos_rates, 99):.3f}")

    print(f"\nPer-user size-1 window share:")
    print(f"  Median: {np.median(size_1_shares):.3f}")
    print(f"  p90:    {np.percentile(size_1_shares, 90):.3f}")
    print(f"  p99:    {np.percentile(size_1_shares, 99):.3f}")
    high_size_1_users = sum(1 for s in size_1_shares if s > 0.5)
    print(f"  Users with >50% size-1 windows: {high_size_1_users:,} ({100*high_size_1_users/len(user_stats):.1f}%)")

    print(f"\nPer-user median window size (Rule 2):")
    print(f"  Median across users: {np.median(median_windows):.1f}")
    print(f"  p90:                 {np.percentile(median_windows, 90):.1f}")


# ============================================================================
# VERDICT
# ============================================================================

def print_verdict() -> None:
    """Final assessment."""
    print("\n" + "=" * 80)
    print("VERDICT & RECOMMENDATION")
    print("=" * 80)

    print("""
Based on the six diagnostics above, assess Tenrec viability:

Path A: DROP entirely (if huge windows, >50% size-1, no category coherence)
  → Tenrec is a sequential engagement dataset, not a menu-choice benchmark

Path B: KEEP only as exploratory appendix (if some structure but significant pathology)
  → Reframe as "sequential exposure consistency" or "recommendation engagement"
  → Do NOT claim main RP benchmark status

Path C: SALVAGE with new construction (if tight local windows, coherent categories)
  → Define small exposure-based windows (last K items before positive action)
  → Use in main benchmark if passing all thresholds

Key pass/fail thresholds:
  1. Median window size: should be < 10 (ideally < 6-7)
  2. p99 window size: should be < 50
  3. Size-1 prevalence: should be < 30%
  4. Chosen-last share: should be < 70%
  5. Category coherence (top-cat share): should be > 0.4-0.5
  6. Pure exposure rows: should be > 20% (true negatives exist)

If 4+ thresholds pass: Path C (salvageable)
If 2-3 thresholds pass: Path B (appendix only)
If < 2 thresholds pass: Path A (drop)
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all six diagnostics."""
    csv_path = Path.home() / ".prefgraph" / "data" / "tenrec" / "QB-video.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        print("Download from Tencent: https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("TENREC EDA: Menu-Choice Viability Assessment (500K sample)")
    print("=" * 80)

    df = load_tenrec_data(csv_path, use_sample=True)

    diagnostic_1_action_consistency(df)
    results_d2 = diagnostic_2_window_sizes(df)
    diagnostic_3_category_coherence(df)
    diagnostic_4_chosen_position(df)
    diagnostic_5_watch_time(df)
    diagnostic_6_user_pathology(df)

    print_verdict()

    print("\n" + "=" * 80)
    print("END EDA")
    print("=" * 80)


if __name__ == "__main__":
    main()
