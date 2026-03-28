#!/usr/bin/env python3
"""Taobao 2×2 salvage grid: item vs category, pv-only vs pv+cart+fav.

All four constructions share:
  - 30-min gap sessions, corrupted timestamps dropped
  - Truncated at first buy (pre-buy events only)
  - No phantom insertion — bought object must appear in observed pre-buy menu
  - Menu size [2, 30]

Varied across a 2×2 grid:
  menu_object:  item | category
  intent:       pv only | pv + cart + fav

For each cell report:
  - Surviving sessions
  - Users with ≥1 / ≥3 / ≥5 clean sessions
  - Median menu size
  - Median sessions per user
  - Mean pairwise comparison opportunities per user (T*(T-1)/2)

Run: python examples/eda/taobao_salvage_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import polars as pl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA_PATH = Path.home() / ".prefgraph" / "data" / "taobao" / "UserBehavior.csv"
SAMPLE_ROWS = 20_000_000
SESSION_GAP = 1800
MENU_SIZE_LO = 2
MENU_SIZE_HI = 30
TS_VALID_LO = 1509494400  # 2017-11-01
TS_VALID_HI = 1514764800  # 2018-01-01


# ============================================================================
# LOAD & CLEAN TIMESTAMPS
# ============================================================================

def load_clean() -> pl.DataFrame:
    print(f"  Loading {SAMPLE_ROWS:,} rows and dropping corrupted timestamps...")
    df = pl.read_csv(
        DATA_PATH,
        has_header=False,
        new_columns=["user_id", "item_id", "category_id", "behavior_type", "timestamp"],
        n_rows=SAMPLE_ROWS,
        schema_overrides={
            "user_id": pl.Int64, "item_id": pl.Int64, "category_id": pl.Int64,
            "behavior_type": pl.Utf8, "timestamp": pl.Int64,
        },
    )
    df = df.filter(
        (pl.col("timestamp") >= TS_VALID_LO) & (pl.col("timestamp") <= TS_VALID_HI)
    )
    print(f"  Clean events: {len(df):,} ({df['user_id'].n_unique():,} users)")
    return df


# ============================================================================
# SESSION ASSIGNMENT
# ============================================================================

def assign_sessions(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort by (user, timestamp) and assign 30-min gap session IDs.
    Returns arrays: user_id, item_id, category_id, behavior_type, timestamp, session_id.
    """
    df_sorted = df.sort(["user_id", "timestamp"])
    user_arr = df_sorted["user_id"].to_numpy()
    item_arr = df_sorted["item_id"].to_numpy()
    cat_arr = df_sorted["category_id"].to_numpy()
    btype_arr = df_sorted["behavior_type"].to_numpy()
    ts_arr = df_sorted["timestamp"].to_numpy()

    new_sess = np.zeros(len(user_arr), dtype=bool)
    new_sess[0] = True
    new_sess[1:] = (user_arr[1:] != user_arr[:-1]) | (np.diff(ts_arr) > SESSION_GAP)
    sess_arr = np.cumsum(new_sess)

    return user_arr, item_arr, cat_arr, btype_arr, ts_arr, sess_arr


# ============================================================================
# CORE SALVAGE FUNCTION
# ============================================================================

def run_construction(
    user_arr, item_arr, cat_arr, btype_arr, ts_arr, sess_arr,
    menu_object: str,   # "item" | "category"
    intent: str,        # "pv" | "pv+cart+fav"
) -> dict:
    """
    For each session:
      1. Find first buy event timestamp and bought item/category.
      2. Collect pre-buy intent events (based on intent parameter).
      3. Build menu as set of items or categories seen pre-buy.
      4. Require bought object in menu (no phantom insertion).
      5. Apply menu size [MENU_SIZE_LO, MENU_SIZE_HI].

    Returns summary statistics.
    """
    # Define which behavior types count as intent evidence
    if intent == "pv":
        intent_types = {"pv"}
    else:
        intent_types = {"pv", "cart", "fav"}

    # Group events by session
    # For memory efficiency, iterate once through the sorted arrays
    n = len(user_arr)
    sessions: dict[int, dict] = {}
    cur_sess = -1

    for i in range(n):
        sid = sess_arr[i]
        if sid != cur_sess:
            cur_sess = sid
            sessions[sid] = {"user": user_arr[i], "buy_ts": None, "buy_item": None,
                             "buy_cat": None, "pre_intent": []}

        btype = btype_arr[i]
        ts = ts_arr[i]
        s = sessions[sid]

        if btype == "buy":
            # Only record the FIRST buy
            if s["buy_ts"] is None:
                s["buy_ts"] = ts
                s["buy_item"] = item_arr[i]
                s["buy_cat"] = cat_arr[i]
        elif btype in intent_types:
            # Will filter to pre-buy later
            s["pre_intent"].append((ts, item_arr[i], cat_arr[i]))

    # Apply strict rules and collect results
    results = []  # list of (user_id, menu_size)
    stats = {
        "total": len(sessions),
        "drop_no_buy": 0,
        "drop_no_prebuy_intent": 0,
        "drop_object_not_in_menu": 0,
        "drop_menu_size": 0,
    }

    for s in sessions.values():
        if s["buy_ts"] is None:
            stats["drop_no_buy"] += 1
            continue

        buy_ts = s["buy_ts"]

        # Pre-buy intent events only
        pre = [(ts, item, cat) for ts, item, cat in s["pre_intent"] if ts < buy_ts]

        if not pre:
            stats["drop_no_prebuy_intent"] += 1
            continue

        # Build menu depending on menu_object
        if menu_object == "item":
            menu_set = {item for _, item, _ in pre}
            bought_obj = s["buy_item"]
        else:  # category
            menu_set = {cat for _, _, cat in pre}
            bought_obj = s["buy_cat"]

        # Require bought object was in observed menu
        if bought_obj not in menu_set:
            stats["drop_object_not_in_menu"] += 1
            continue

        # Menu size filter
        menu_size = len(menu_set)
        if menu_size < MENU_SIZE_LO or menu_size > MENU_SIZE_HI:
            stats["drop_menu_size"] += 1
            continue

        results.append((s["user"], menu_size))

    # Aggregate user-level statistics
    user_sessions: dict[int, list[int]] = defaultdict(list)
    for uid, ms in results:
        user_sessions[uid].append(ms)

    n_sessions = len(results)
    n_users_any = len(user_sessions)
    n_users_3 = sum(1 for v in user_sessions.values() if len(v) >= 3)
    n_users_5 = sum(1 for v in user_sessions.values() if len(v) >= 5)

    sessions_per_user = np.array([len(v) for v in user_sessions.values()]) if user_sessions else np.array([0])
    menu_sizes = np.array([ms for _, ms in results]) if results else np.array([0])

    median_sessions = float(np.median(sessions_per_user))
    median_menu = float(np.median(menu_sizes))
    mean_pairwise = float(np.mean(sessions_per_user * (sessions_per_user - 1) / 2))

    return {
        "menu_object": menu_object,
        "intent": intent,
        "n_sessions": n_sessions,
        "n_users_any": n_users_any,
        "n_users_3": n_users_3,
        "n_users_5": n_users_5,
        "median_sessions_per_user": median_sessions,
        "median_menu_size": median_menu,
        "mean_pairwise": mean_pairwise,
        "sessions_per_user": sessions_per_user,
        "menu_sizes": menu_sizes,
        "stats": stats,
    }


# ============================================================================
# PRINT RESULTS
# ============================================================================

def print_cell(r: dict, total_sessions: int) -> None:
    label = f"[{r['menu_object'].upper()} | {r['intent'].upper()}]"
    print(f"\n  {label}")
    print(f"    Surviving sessions:          {r['n_sessions']:>8,}  ({r['n_sessions']/total_sessions*100:.2f}% of total)")
    print(f"    Users ≥1 clean session:      {r['n_users_any']:>8,}")
    print(f"    Users ≥3 clean sessions:     {r['n_users_3']:>8,}")
    print(f"    Users ≥5 clean sessions:     {r['n_users_5']:>8,}")
    print(f"    Median sessions/user:        {r['median_sessions_per_user']:>8.1f}")
    print(f"    Median menu size:            {r['median_menu_size']:>8.1f}")
    print(f"    Mean pairwise opportunities: {r['mean_pairwise']:>8.2f}")

    s = r["stats"]
    print(f"    --- Dropout ---")
    print(f"    No buy:                   {s['drop_no_buy']:>8,}  ({s['drop_no_buy']/s['total']*100:.1f}%)")
    print(f"    No pre-buy intent event:  {s['drop_no_prebuy_intent']:>8,}  ({s['drop_no_prebuy_intent']/s['total']*100:.1f}%)")
    print(f"    Bought obj not in menu:   {s['drop_object_not_in_menu']:>8,}  ({s['drop_object_not_in_menu']/s['total']*100:.1f}%)")
    print(f"    Menu size outside [2,30]: {s['drop_menu_size']:>8,}  ({s['drop_menu_size']/s['total']*100:.1f}%)")


def print_sessions_per_user_dist(r: dict) -> None:
    label = f"[{r['menu_object'].upper()} | {r['intent'].upper()}]"
    arr = r["sessions_per_user"]
    print(f"\n  {label} — Sessions per user distribution:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    p{p:3d}: {np.percentile(arr, p):.0f}")
    for t in [1, 2, 3, 4, 5, 8, 10]:
        n = int(np.sum(arr >= t))
        pct = n / len(arr) * 100
        print(f"    >= {t:2d} sessions: {n:>7,} users  ({pct:.1f}%)")


def print_menu_size_dist(r: dict) -> None:
    label = f"[{r['menu_object'].upper()} | {r['intent'].upper()}]"
    arr = r["menu_sizes"]
    print(f"\n  {label} — Menu size distribution:")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"    p{p:3d}: {np.percentile(arr, p):.0f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("TAOBAO 2×2 SALVAGE GRID")
    print(f"Sample: first {SAMPLE_ROWS:,} rows")
    print(f"Menu size: [{MENU_SIZE_LO}, {MENU_SIZE_HI}]")
    print("All constructions: pre-first-buy truncation, no phantom insertion")
    print("="*70)

    df = load_clean()

    print("\n  Assigning sessions...")
    user_arr, item_arr, cat_arr, btype_arr, ts_arr, sess_arr = assign_sessions(df)
    total_sessions = int(sess_arr[-1]) if len(sess_arr) > 0 else 0
    print(f"  Total sessions: {total_sessions:,}")

    # Run 2×2 grid
    configs = [
        ("item",     "pv"),
        ("item",     "pv+cart+fav"),
        ("category", "pv"),
        ("category", "pv+cart+fav"),
    ]

    print("\n" + "="*70)
    print("RUNNING 4 CONSTRUCTIONS")
    print("="*70)

    results = []
    for menu_object, intent in configs:
        label = f"[{menu_object.upper()} | {intent.upper()}]"
        print(f"\n  Running {label}...")
        r = run_construction(user_arr, item_arr, cat_arr, btype_arr, ts_arr, sess_arr,
                             menu_object, intent)
        results.append(r)
        print(f"    Sessions: {r['n_sessions']:,}  Users(≥5): {r['n_users_5']:,}")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n  {'Construction':<28} {'Sessions':>10} {'%total':>7} {'Users≥1':>9} {'Users≥3':>9} {'Users≥5':>9} {'Med.sess':>9} {'Med.menu':>9} {'Pairwise':>9}")
    print(f"  {'-'*28} {'-'*10} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for r in results:
        label = f"{r['menu_object']}/{r['intent']}"
        print(f"  {label:<28} {r['n_sessions']:>10,} {r['n_sessions']/total_sessions*100:>6.2f}% "
              f"{r['n_users_any']:>9,} {r['n_users_3']:>9,} {r['n_users_5']:>9,} "
              f"{r['median_sessions_per_user']:>9.1f} {r['median_menu_size']:>9.1f} "
              f"{r['mean_pairwise']:>9.2f}")

    # Detailed per-cell output
    print("\n" + "="*70)
    print("DETAILED RESULTS PER CELL")
    print("="*70)
    for r in results:
        print_cell(r, total_sessions)

    # Session and menu distributions
    print("\n" + "="*70)
    print("SESSION DEPTH AND MENU SIZE DISTRIBUTIONS")
    print("="*70)
    for r in results:
        print_sessions_per_user_dist(r)
        print_menu_size_dist(r)

    # Verdict
    print("\n" + "="*70)
    print("VERDICT BY CELL")
    print("="*70)
    for r in results:
        label = f"[{r['menu_object'].upper()} | {r['intent'].upper()}]"
        u5 = r["n_users_5"]
        u3 = r["n_users_3"]
        med = r["median_sessions_per_user"]
        pw = r["mean_pairwise"]

        if u5 == 0 and u3 < 100:
            verdict = "DEAD — no usable benchmark"
        elif u5 < 200 and med <= 1.0:
            verdict = "DEAD — insufficient session depth"
        elif u5 < 500 and med <= 1.5:
            verdict = "WEAK — appendix only, with heavy caveats"
        elif u5 < 2000 or med < 2.0:
            verdict = "MARGINAL — keep with explicit caveats"
        elif u5 >= 2000 and med >= 2.0 and pw >= 2.0:
            verdict = "VIABLE — suitable for main benchmark"
        else:
            verdict = "MARGINAL — review manually"

        print(f"\n  {label}")
        print(f"    Users≥5={u5}, median sessions={med:.1f}, mean pairwise={pw:.2f}")
        print(f"    → {verdict}")

    print("\nDone.")


if __name__ == "__main__":
    main()
