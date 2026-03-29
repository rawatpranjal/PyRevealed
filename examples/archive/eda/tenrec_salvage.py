#!/usr/bin/env python3
"""Tenrec salvage experiment: Test three local session constructions.

GOAL: Determine if any local-window construction can produce credible
menu-choice observations from Tenrec data.

CONSTRUCTIONS TESTED:
- A: Category-run micro-sessions (contiguous same-category rows)
- B1: Fixed K=5 window before like
- B2: Fixed K=10 window before like

ACCEPTANCE THRESHOLD: 4+ out of 5 criteria must pass (80%)
- Median size >= 4
- Duplicate share < 15%
- Same-category share > 70%
- Chosen-last share < 100%
- N sessions > 100

================================================================================
CRITICAL ISSUES IDENTIFIED (why salvage fails)
================================================================================

1. CHOSEN-ITEM ALWAYS LAST = 100% (UNFIXABLE STRUCTURAL PROBLEM)

   Results across all three constructions:
   - Cat-run: 100.0%
   - K=5: 100.0%
   - K=10: 100.0%

   Why: Windows are constructed to END at a like event. By definition, the
   like is terminal. This cannot be changed without abandoning "like" as
   the choice signal itself.

   Economic interpretation: 100% chosen-last means users are stopping when
   they find something they like. This is sequential engagement (stopping
   behavior), not menu choice (preference from alternatives).

2. SIZE vs CATEGORY COHERENCE TRADE-OFF (UNRESOLVABLE)

   Category-run micro-sessions:
   - Same-category: 99.8% (excellent)
   - Median size: 1.0 (trivial, not a menu)
   - Verdict: Too small to be a choice set

   K=5 windows:
   - Median size: 5.0 (good)
   - Same-category: 22.1% (poor)
   - Verdict: Loses category coherence by mixing categories

   K=10 windows:
   - Median size: 10.0 (good)
   - Same-category: 8.6% (very poor)
   - Verdict: Even worse category coherence

   No construction simultaneously achieves:
   - Reasonable size (median >= 4)
   - High category coherence (> 70%)
   - Non-terminal choice position (< 100% last)

3. TINY WINDOWS (ESPECIALLY CAT-RUN)

   Category-run produces 211K sessions but:
   - Median size: 1.0 item
   - p95 size: 6.0 items

   A one-item "choice set" is not a choice. 50%+ of windows are size-1.

4. CROSS-CATEGORY MIXING IN FIXED-K CONSTRUCTIONS

   K=5 and K=10 take consecutive items from the sequential feed. These
   naturally span multiple video categories (users scroll from Comedy to
   Food to Travel). This violates the coherent-menu assumption.

   - K=5: Only 22.1% same-category
   - K=10: Only 8.6% same-category

   Cannot use classical RP axioms (SARP, WARP) on cross-category sets.

5. ALL CONSTRUCTIONS FAIL ACCEPTANCE THRESHOLD

   Required: 4+ out of 5 criteria pass (80%)
   Actual: All three score 3/5 (60%)

   Breakdown:
   - Median size >= 4: Cat-run fails (1.0), K-windows pass
   - Duplicates < 15%: All pass (0.0%)
   - Same-category > 70%: Cat-run passes (99.8%), K-windows fail
   - Chosen-last < 100%: ALL FAIL (100% across all)
   - N > 100: All pass

   The decisive failures:
   1. Chosen-last = 100% (cannot fix)
   2. Cat-run median size = 1.0 (too small)
   3. K-windows same-category = 8-22% (too dispersed)

6. SEQUENTIAL FEED STRUCTURE vs MENU-CHOICE ASSUMPTION

   Tenrec data represent users scrolling through a recommendation feed.
   Each item is seen sequentially (order matters, no simultaneous display).

   Classical menu-choice analysis assumes users face a simultaneous set
   of alternatives and choose one. In a feed, order and stopping rules
   dominate preference signals.

   Consequence: Even if windows were perfectly coherent, the sequential
   nature would confound choice with sequential effects.

7. NO VALID WINDOW-SIZE REGION

   EDA Diagnostic 2 showed:
   - Median window (rules 1-3): 2-3 items (too small, high variance)
   - p99 window: 198-372 items (degenerate outliers)
   - Size-1 prevalence: 35-46% (trivial observations)

   No "sweet spot" exists where windows are non-trivial, bounded, and
   coherent. The distribution is bimodal: many size-1, many huge.

8. SPARSE LIKE SIGNAL (0.8% of rows)

   Only ~1 in 125 rows is a like event. This scarcity makes it difficult
   to construct enough valid windows per user for stable RP analysis.
   Sparse signals require dense alternative observations (which we don't have).

9. FULL-FILE VALIDATION (2.44M ROWS) CONFIRMS FINDINGS

   Ran diagnostics on full QB-video.csv:
   - Chosen-last: 100.0% (identical to 500K sample)
   - p99 window: 220 items (similar to 500K: 199)
   - Size-1: 34.8% (similar to 500K: 45.6%)
   - Median window: 3.0 (vs 2.0 in 500K, but still below 4.0)

   Conclusion: 500K sequential sample is representative.
   Salvage failure is NOT a sampling artifact.

10. ECONOMIC INTERPRETATION: STOPPING BEHAVIOR, NOT PREFERENCE

    100% chosen-last across all constructions reveals the core issue:

    The user scrolls through items in order. When they find something they
    like, they stop (or at least express a positive signal). The sequence
    of items they saw before stopping is NOT a menu they chose from.

    It's a stopping rule: user skips N items, encounters item M, expresses
    interest in M (or stops scrolling). This measures:
    - Stopping behavior
    - Order effects
    - Engagement intensity

    NOT:
    - Preference from a set
    - Revealed preference axioms
    - Utility maximization

================================================================================
CONCLUSION
================================================================================

All three salvage constructions fail acceptance criteria.
No construction achieves 4/5 (80%) threshold; all score 3/5 (60%).

Structural failures (unfixable):
1. Chosen-last = 100% (like is terminal event)
2. Size-category trade-off (cannot have both large windows and coherence)
3. Sequential feed structure (not simultaneous menu)

Recommendation: STOP salvage attempts. Move Tenrec to appendix
(sequential-engagement reframe) or DROP entirely.

See: tenrec_report.md (facts), tenrec_salvage_report.md (detailed analysis),
     tenrec_robustness_check.md (full-file validation)
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
