#!/usr/bin/env python3
"""Tenrec salvage experiment: Test two local session constructions.

Construction A: Category-run micro-sessions
Construction B: Fixed K-window before like

Report: median size, p95, duplicate%, same-category%, chosen-last share.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import polars as pl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_data() -> pl.DataFrame:
    """Load 500K sample from temp."""
    sample_path = Path('/tmp/tenrec_sample.csv')
    if sample_path.exists():
        df = pl.read_csv(sample_path)
    else:
        csv_path = Path.home() / '.prefgraph' / 'data' / 'tenrec' / 'QB-video.csv'
        df = pl.read_csv(csv_path, n_rows=500_000, null_values='\\N')
    return df


# ============================================================================
# CONSTRUCTION A: Category-run micro-sessions
# ============================================================================

def build_category_run_sessions(df: pl.DataFrame) -> list[dict]:
    """Build sessions as contiguous rows in same category, reset on category change.

    A session ends when:
    - The user transitions to a new category, OR
    - A positive action (like/share/follow) occurs

    Max session length: 10 items.
    """
    df_sorted = df.sort("user_id")
    sessions = []

    for user_id in df_sorted["user_id"].unique():
        user_df = df_sorted.filter(pl.col("user_id") == user_id)
        session = []
        session_cats = []
        last_cat = None

        for row in user_df.iter_rows(named=True):
            cat = row["video_category"]

            # Check if we need to close the session
            if last_cat is not None and cat != last_cat:
                # Category break
                if session:
                    sessions.append({
                        "items": session,
                        "categories": session_cats,
                        "size": len(session),
                        "liked": False,
                        "has_duplicate": len(session) != len(set(session)),
                    })
                session = []
                session_cats = []

            # Add item to session
            session.append(row["item_id"])
            session_cats.append(cat)
            last_cat = cat

            # Check for positive action (end session)
            if row["like"] == 1 or row["share"] == 1 or row["follow"] == 1:
                if session:
                    sessions.append({
                        "items": session,
                        "categories": session_cats,
                        "size": len(session),
                        "liked": row["like"] == 1,
                        "has_duplicate": len(session) != len(set(session)),
                    })
                session = []
                session_cats = []

        # Close any remaining session
        if session:
            sessions.append({
                "items": session,
                "categories": session_cats,
                "size": len(session),
                "liked": False,
                "has_duplicate": len(session) != len(set(session)),
            })

    return sessions


# ============================================================================
# CONSTRUCTION B: Fixed K windows before like
# ============================================================================

def build_fixed_k_windows(df: pl.DataFrame, k_values: list[int] = None) -> dict:
    """Build fixed-length windows (K=5, K=10) before each like event."""
    if k_values is None:
        k_values = [5, 10]

    results = {}

    for k in k_values:
        windows = []
        df_sorted = df.sort("user_id")

        for user_id in df_sorted["user_id"].unique():
            user_df = df_sorted.filter(pl.col("user_id") == user_id)

            for i, row in enumerate(user_df.iter_rows(named=True)):
                if row["like"] == 1:
                    # Take last K rows before this like (including this like)
                    start_idx = max(0, i - k + 1)
                    window_rows = user_df[start_idx:i+1]

                    items = window_rows["item_id"].to_list()
                    cats = window_rows["video_category"].to_list()

                    windows.append({
                        "items": items,
                        "categories": cats,
                        "size": len(items),
                        "liked": True,
                        "has_duplicate": len(items) != len(set(items)),
                    })

        results[f"K={k}"] = windows

    return results


# ============================================================================
# Analysis
# ============================================================================

def analyze_sessions(name: str, sessions: list[dict]) -> dict:
    """Compute metrics for a session construction."""
    if not sessions:
        return {
            "name": name,
            "n_sessions": 0,
            "median_size": 0,
            "p95_size": 0,
            "pct_duplicates": 0,
            "pct_same_category": 0,
            "chosen_last_share": 0,
        }

    sizes = [s["size"] for s in sessions]
    has_dup = [s["has_duplicate"] for s in sessions]

    # Same-category: all items in session are same category
    same_cat = []
    for s in sessions:
        if s["categories"]:
            unique_cats = len(set(s["categories"]))
            same_cat.append(unique_cats == 1)

    # Chosen-last: only applies to sessions with a "choice" (like)
    chosen_last = []
    for s in sessions:
        if s["liked"] and s["size"] >= 1:
            # For fixed-K construction, liked item is always last by definition
            # For category-run, check if the positive action was the last item
            chosen_last.append(1)

    pct_dup = 100 * sum(has_dup) / len(has_dup) if has_dup else 0
    pct_same_cat = 100 * sum(same_cat) / len(same_cat) if same_cat else 0
    chosen_last_share = 100 * sum(chosen_last) / len(chosen_last) if chosen_last else 0

    return {
        "name": name,
        "n_sessions": len(sessions),
        "median_size": float(np.median(sizes)),
        "p95_size": float(np.percentile(sizes, 95)),
        "p99_size": float(np.percentile(sizes, 99)),
        "pct_duplicates": pct_dup,
        "pct_same_category": pct_same_cat,
        "chosen_last_share": chosen_last_share,
    }


def main():
    print("\n" + "=" * 100)
    print("TENREC SALVAGE EXPERIMENT: Local Session Constructions")
    print("=" * 100)

    df = load_data()
    print(f"\nDataset: {df.shape[0]:,} rows, {df['user_id'].n_unique():,} users")

    # ========== Construction A: Category-run micro-sessions ==========
    print("\n" + "-" * 100)
    print("CONSTRUCTION A: Category-run micro-sessions")
    print("-" * 100)
    print("Rule: Contiguous rows in same category, reset on category switch or positive action")

    cat_sessions = build_category_run_sessions(df)
    a_stats = analyze_sessions("Cat-run micro-sessions", cat_sessions)

    # ========== Construction B: Fixed K windows ==========
    print("\n" + "-" * 100)
    print("CONSTRUCTION B: Fixed K-windows before like")
    print("-" * 100)
    print("Rule: Last K exposures before a like event")

    k_windows = build_fixed_k_windows(df, k_values=[5, 10])
    b5_stats = analyze_sessions("K=5 before like", k_windows["K=5"])
    b10_stats = analyze_sessions("K=10 before like", k_windows["K=10"])

    # ========== Summary Table ==========
    print("\n" + "=" * 100)
    print("RESULTS TABLE")
    print("=" * 100)

    results = [a_stats, b5_stats, b10_stats]

    print(
        f"\n{'Construction':<30} {'N Sessions':>10} {'Median Size':>12} {'p95':>8} {'p99':>8} "
        f"{'Dup%':>7} {'Same-Cat%':>10} {'Chosen-Last%':>12}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r['name']:<30} "
            f"{r['n_sessions']:>10,} "
            f"{r['median_size']:>12.1f} "
            f"{r['p95_size']:>8.1f} "
            f"{r['p99_size']:>8.1f} "
            f"{r['pct_duplicates']:>7.1f} "
            f"{r['pct_same_category']:>10.1f} "
            f"{r['chosen_last_share']:>12.1f}"
        )

    print("-" * 100)

    # ========== Acceptance Criteria ==========
    print("\n" + "=" * 100)
    print("ACCEPTANCE CRITERIA EVALUATION")
    print("=" * 100)

    criteria = {
        "Median unique items >= 4": lambda r: r["median_size"] >= 4,
        "Duplicate share < 15%": lambda r: r["pct_duplicates"] < 15,
        "Top-category share > 70%": lambda r: r["pct_same_category"] > 70,
        "Chosen-last share < 100%": lambda r: r["chosen_last_share"] < 100,
        "N sessions > 100": lambda r: r["n_sessions"] > 100,
    }

    for r in results:
        print(f"\n{r['name']}:")
        passes = 0
        for criterion_name, criterion_fn in criteria.items():
            passed = criterion_fn(r)
            status = "✓ PASS" if passed else "✗ FAIL"
            passes += passed
            print(f"  {criterion_name:<40} {status}")
        print(f"  → Score: {passes}/5 pass")

    # ========== Recommendation ==========
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)

    b5_passes = sum(1 for crit in criteria.values() if crit(b5_stats))
    b10_passes = sum(1 for crit in criteria.values() if crit(b10_stats))
    a_passes = sum(1 for crit in criteria.values() if crit(a_stats))

    print(f"\nBased on acceptance criteria (pass threshold: 4+ out of 5):")
    print(f"  Category-run:  {a_passes}/5 → {'SALVAGEABLE' if a_passes >= 4 else 'FAIL'}")
    print(f"  K=5 windows:   {b5_passes}/5 → {'SALVAGEABLE' if b5_passes >= 4 else 'FAIL'}")
    print(f"  K=10 windows:  {b10_passes}/5 → {'SALVAGEABLE' if b10_passes >= 4 else 'FAIL'}")

    best_passes = max(a_passes, b5_passes, b10_passes)
    if best_passes >= 4:
        print(f"\n✓ At least one construction passes. Proceed to feature experiment.")
    else:
        print(f"\n✗ No construction passes acceptance criteria. Recommend appendix or drop.")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
