#!/usr/bin/env python3
"""Taobao User Behavior EDA: Auditing menu-choice construction assumptions.

Audits each assumption made when converting raw Taobao events into
MenuChoiceLog objects for revealed preference analysis.

Raw format (no header):
  user_id, item_id, category_id, behavior_type, timestamp
  behavior_type in {pv (pageview), buy, cart, fav}

The loader (_taobao.py) makes these assumptions:
  A1. Sessions defined by 30-min (1800s) inactivity gap between events of the same user.
  A2. Menu = set of viewed items (pv) in a session.
  A3. Choice = the purchased item (buy) in that session.
  A4. Only sessions with exactly 1 unique purchased item are valid.
  A5. If the purchased item was not viewed before buying, it is inserted into the menu.
  A6. Menu size must be in [2, 50] - singletons dropped, huge menus dropped.
  A7. Users with < 5 valid sessions are excluded.
  A8. Top 50K users by session count are taken (sort by activity descending).

Nine diagnostics auditing these in turn:
  D1. Raw data overview (behavior mix, time range, user counts)
  D2. Inter-event gap distribution - is 30 min the right threshold? (A1)
  D3. Session structure - sessions per user, events per session (A1, A7, A8)
  D4. Purchase-per-session distribution - 0/1/2+ buys (A3, A4)
  D5. Purchase-in-menu rate - how often was the bought item viewed? (A2, A5)
  D6. Purchase temporal position - is the purchase always the LAST viewed item? (A3)
  D7. Menu size distribution - before/after filters, filter dropout rates (A6)
  D8. Category coherence - within-session category consistency (A2 quality)
  D9. User qualification sensitivity - how much does min_sessions threshold matter? (A7)

Run: python examples/eda/taobao_eda.py

Sample: reads first 20M rows (~20% of the 100M row dataset) for speed.
Findings are written to taobao_eda_report.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import polars as pl
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA_PATH = Path.home() / ".prefgraph" / "data" / "taobao" / "UserBehavior.csv"
SESSION_GAP = 1800   # 30 minutes (seconds) - the loader's threshold
SAMPLE_ROWS = 20_000_000  # 20M / 100M = 20% of data


# ============================================================================
# DATA LOADING
# ============================================================================

def load_sample() -> pl.DataFrame:
    """Load first SAMPLE_ROWS rows of UserBehavior.csv (no header)."""
    print(f"  Loading {SAMPLE_ROWS:,} rows from {DATA_PATH}...")
    df = pl.read_csv(
        DATA_PATH,
        has_header=False,
        new_columns=["user_id", "item_id", "category_id", "behavior_type", "timestamp"],
        n_rows=SAMPLE_ROWS,
        dtypes={
            "user_id": pl.Int64,
            "item_id": pl.Int64,
            "category_id": pl.Int64,
            "behavior_type": pl.Utf8,
            "timestamp": pl.Int64,
        },
    )
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ============================================================================
# DIAGNOSTIC 1: Raw data overview
# ============================================================================

def diagnostic_1_overview(df: pl.DataFrame) -> dict:
    """Behavior mix, time range, unique users/items."""
    print("\n" + "="*70)
    print("D1. RAW DATA OVERVIEW")
    print("="*70)

    n_rows = len(df)
    n_users = df["user_id"].n_unique()
    n_items = df["item_id"].n_unique()
    n_categories = df["category_id"].n_unique()

    # Behavior type distribution
    btype_counts = (
        df.group_by("behavior_type")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / n_rows * 100).alias("pct"))
        .sort("count", descending=True)
    )
    print(f"\n  Total rows: {n_rows:,}")
    print(f"  Unique users: {n_users:,}")
    print(f"  Unique items: {n_items:,}")
    print(f"  Unique categories: {n_categories:,}")

    # Time range
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    from datetime import datetime, timezone
    dt_min = datetime.fromtimestamp(ts_min, tz=timezone.utc)
    dt_max = datetime.fromtimestamp(ts_max, tz=timezone.utc)
    print(f"\n  Time range: {dt_min.strftime('%Y-%m-%d')} to {dt_max.strftime('%Y-%m-%d')}")
    print(f"  Span: {(ts_max - ts_min) / 86400:.1f} days")

    print(f"\n  Behavior type counts:")
    for row in btype_counts.iter_rows(named=True):
        print(f"    {row['behavior_type']:<8} {row['count']:>10,}  ({row['pct']:.1f}%)")

    return {
        "n_rows": n_rows, "n_users": n_users, "n_items": n_items,
        "n_categories": n_categories, "ts_min": ts_min, "ts_max": ts_max,
        "btype_counts": {r["behavior_type"]: r["count"] for r in btype_counts.iter_rows(named=True)},
    }


# ============================================================================
# DIAGNOSTIC 2: Inter-event gap distribution (assumption A1)
# ============================================================================

def diagnostic_2_gap_distribution(df: pl.DataFrame) -> dict:
    """
    A1: Sessions defined by 30-min gap.

    What fraction of inter-event gaps fall below various thresholds?
    Is 30 min a natural break or arbitrary? How many sessions result
    from different threshold choices?

    Key concern: if gaps cluster just below and above 30 min, the threshold
    is knife-edge and session assignments are unstable.
    """
    print("\n" + "="*70)
    print("D2. INTER-EVENT GAP DISTRIBUTION (assumption: 30-min session boundary)")
    print("="*70)

    # Sort by (user, timestamp), compute gaps within each user
    df_sorted = df.sort(["user_id", "timestamp"])

    # Lag within user
    gaps = (
        df_sorted
        .with_columns([
            pl.col("user_id").shift(1).alias("prev_user"),
            pl.col("timestamp").shift(1).alias("prev_ts"),
        ])
        .filter(pl.col("user_id") == pl.col("prev_user"))  # same user only
        .with_columns((pl.col("timestamp") - pl.col("prev_ts")).alias("gap_s"))
        .filter(pl.col("gap_s") >= 0)  # sanity: no negative gaps
        .select("gap_s")
    )

    gap_arr = gaps["gap_s"].to_numpy()
    n_gaps = len(gap_arr)

    print(f"\n  Total within-user consecutive gaps: {n_gaps:,}")

    # Percentile breakdown
    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]
    print(f"\n  Gap distribution (seconds):")
    for p in pcts:
        v = np.percentile(gap_arr, p)
        label = f"{v:.0f}s"
        if v < 60:
            human = f"{v:.0f}s"
        elif v < 3600:
            human = f"{v/60:.1f}min"
        elif v < 86400:
            human = f"{v/3600:.1f}hr"
        else:
            human = f"{v/86400:.1f}days"
        print(f"    p{p:5.1f}: {label:>10}  ({human})")

    # Fraction below common thresholds
    print(f"\n  Fraction of gaps BELOW threshold (= would NOT start a new session):")
    thresholds = [60, 300, 900, 1800, 3600, 7200, 14400, 86400]
    threshold_fracs = {}
    for t in thresholds:
        frac = np.mean(gap_arr < t)
        threshold_fracs[t] = frac
        label = f"{t}s"
        if t < 3600:
            label = f"{t//60}min"
        elif t < 86400:
            label = f"{t//3600}hr"
        else:
            label = f"{t//86400}d"
        print(f"    < {label:<6}: {frac*100:.1f}%  (these gaps stay within same session)")

    # Session count sensitivity to threshold
    print(f"\n  Session count sensitivity (how many sessions under different gaps):")
    for t in [600, 900, 1800, 3600, 7200]:
        n_new_sessions = int(np.sum(gap_arr >= t))
        label = f"{t//60}min"
        print(f"    {label:<8} gap: {n_new_sessions:>8,} session breaks  "
              f"(+{(n_new_sessions - np.sum(gap_arr >= SESSION_GAP))/(np.sum(gap_arr >= SESSION_GAP)+1)*100:+.1f}% vs 30min)")

    # Histogram of gaps in minutes (0-120 min range) to spot clusters
    print(f"\n  Gap histogram [0-120 min] (looking for bimodal structure):")
    bins = [0, 1, 5, 10, 15, 20, 25, 30, 40, 60, 90, 120]  # minutes
    for i in range(len(bins)-1):
        lo, hi = bins[i]*60, bins[i+1]*60
        count = int(np.sum((gap_arr >= lo) & (gap_arr < hi)))
        pct = count / n_gaps * 100
        bar = "#" * int(pct / 0.5)  # 1 bar = 0.5%
        print(f"    [{bins[i]:3d}-{bins[i+1]:3d}min]: {count:>8,}  ({pct:5.1f}%)  {bar}")

    return {"gap_arr": gap_arr, "n_gaps": n_gaps, "threshold_fracs": threshold_fracs}


# ============================================================================
# DIAGNOSTIC 3: Session structure (assumptions A1, A7, A8)
# ============================================================================

def build_sessions(df: pl.DataFrame, gap_threshold: int = SESSION_GAP) -> pl.DataFrame:
    """
    Assign session IDs using gap-based rule.
    Returns df with session_id column added.
    """
    df_sorted = df.sort(["user_id", "timestamp"])

    # Vectorised gap computation
    user_arr = df_sorted["user_id"].to_numpy()
    ts_arr = df_sorted["timestamp"].to_numpy()

    new_session = np.zeros(len(user_arr), dtype=bool)
    new_session[0] = True
    new_user = user_arr[1:] != user_arr[:-1]
    large_gap = np.diff(ts_arr) > gap_threshold
    new_session[1:] = new_user | large_gap

    session_ids = np.cumsum(new_session)
    return df_sorted.with_columns(pl.Series("session_id", session_ids))


def diagnostic_3_session_structure(df: pl.DataFrame) -> dict:
    """
    Sessions per user distribution, events per session distribution.
    Does 30-min gap produce reasonable session counts?
    """
    print("\n" + "="*70)
    print("D3. SESSION STRUCTURE (30-min gap)")
    print("="*70)

    df_sess = build_sessions(df)

    # Events per session (all behavior types)
    events_per_session = (
        df_sess.group_by("session_id")
        .agg(pl.len().alias("n_events"))
        ["n_events"].to_numpy()
    )

    # Sessions per user
    sessions_per_user = (
        df_sess.group_by("user_id")
        .agg(pl.col("session_id").n_unique().alias("n_sessions"))
        ["n_sessions"].to_numpy()
    )

    n_sessions_total = df_sess["session_id"].n_unique()
    n_users = df_sess["user_id"].n_unique()

    print(f"\n  Total sessions (30-min gap): {n_sessions_total:,}")
    print(f"  Unique users: {n_users:,}")
    print(f"  Mean sessions/user: {n_sessions_total/n_users:.1f}")

    print(f"\n  Events per session:")
    for p in [0, 5, 25, 50, 75, 90, 95, 99, 100]:
        v = np.percentile(events_per_session, p)
        print(f"    p{p:3d}: {v:.0f}")

    print(f"\n  Sessions per user:")
    for p in [0, 5, 25, 50, 75, 90, 95, 99, 100]:
        v = np.percentile(sessions_per_user, p)
        print(f"    p{p:3d}: {v:.0f}")

    # Sessions per user histogram
    print(f"\n  Users by session count (A7: min_sessions=5 filter):")
    bins = [1, 2, 3, 4, 5, 10, 20, 50, 100, 10000]
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        count = int(np.sum((sessions_per_user >= lo) & (sessions_per_user < hi)))
        pct = count / len(sessions_per_user) * 100
        print(f"    [{lo:4d}-{hi:4d}): {count:>7,} users  ({pct:.1f}%)")

    # Sensitivity to min_sessions threshold
    print(f"\n  Qualifying users by min_sessions threshold:")
    for min_s in [1, 3, 5, 10, 15, 20, 30]:
        n_qual = int(np.sum(sessions_per_user >= min_s))
        pct = n_qual / len(sessions_per_user) * 100
        print(f"    >= {min_s:2d} sessions: {n_qual:>7,} users  ({pct:.1f}%)")

    return {"df_sess": df_sess, "events_per_session": events_per_session,
            "sessions_per_user": sessions_per_user}


# ============================================================================
# DIAGNOSTIC 4: Purchase-per-session distribution (assumptions A3, A4)
# ============================================================================

def diagnostic_4_purchase_per_session(df_sess: pl.DataFrame) -> dict:
    """
    A3/A4: Only sessions with exactly 1 unique purchased item are valid.

    What fraction of sessions have 0, 1, 2+ purchases?
    How many sessions are dropped by the exactly-1-buy filter?
    """
    print("\n" + "="*70)
    print("D4. PURCHASE-PER-SESSION DISTRIBUTION (assumption: exactly 1 buy/session)")
    print("="*70)

    buys = df_sess.filter(pl.col("behavior_type") == "buy")
    views = df_sess.filter(pl.col("behavior_type") == "pv")

    n_sessions_total = df_sess["session_id"].n_unique()

    # Buys per session
    buys_per_session = (
        buys.group_by("session_id")
        .agg(pl.col("item_id").n_unique().alias("n_unique_buys"))
    )

    # Join with all sessions (sessions without any buy get 0)
    all_sessions = df_sess.select("session_id").unique()
    buys_per_session_full = (
        all_sessions
        .join(buys_per_session, on="session_id", how="left")
        .with_columns(pl.col("n_unique_buys").fill_null(0))
    )

    counts = buys_per_session_full["n_unique_buys"].value_counts().sort("n_unique_buys")

    print(f"\n  Total sessions: {n_sessions_total:,}")
    print(f"\n  Distribution of unique bought items per session:")
    total_with_pv = views["session_id"].n_unique()
    print(f"  (Sessions with at least 1 pv: {total_with_pv:,})")
    for row in counts.iter_rows(named=True):
        n = row["n_unique_buys"]
        c = row["count"]
        pct = c / n_sessions_total * 100
        flag = " <-- KEPT (A4)" if n == 1 else (" <-- DROPPED (0 buys)" if n == 0 else " <-- DROPPED (multi-buy)")
        print(f"    {n} buys: {c:>8,}  ({pct:.1f}%){flag}")

    # Sessions with exactly 1 buy AND at least 1 pv (valid sessions before menu-size filter)
    valid_session_ids = set(
        buys_per_session.filter(pl.col("n_unique_buys") == 1)["session_id"].to_list()
    )
    n_valid = len(valid_session_ids)
    pct_valid = n_valid / n_sessions_total * 100
    print(f"\n  Sessions with exactly 1 unique buy: {n_valid:,}  ({pct_valid:.1f}% of all sessions)")

    return {"valid_session_ids": valid_session_ids,
            "buys_per_session": buys_per_session,
            "buys": buys,
            "views": views}


# ============================================================================
# DIAGNOSTIC 5: Purchase-in-menu rate (assumptions A2, A5)
# ============================================================================

def diagnostic_5_purchase_in_menu(
    df_sess: pl.DataFrame,
    valid_session_ids: set,
    buys: pl.DataFrame,
    views: pl.DataFrame,
) -> dict:
    """
    A2/A5: The loader does `menu | {choice}` - inserts the purchased item
    into the menu even if it was never viewed.

    How often does a user buy something they never viewed in the same session?
    This determines how artificial the menu is.

    If the purchase was NOT viewed, the "menu" has a phantom item inserted.
    The choice set is then partly observed, partly inferred.
    """
    print("\n" + "="*70)
    print("D5. PURCHASE-IN-MENU RATE (assumption: buy item added to menu if not viewed)")
    print("="*70)

    # Restrict to valid sessions (exactly 1 buy)
    valid_buys = buys.filter(pl.col("session_id").is_in(list(valid_session_ids)))

    # One buy per valid session - pick the purchased item
    session_buy_item = (
        valid_buys.group_by("session_id")
        .agg(pl.col("item_id").first().alias("bought_item"))
    )

    # Viewed items per valid session
    valid_views = views.filter(pl.col("session_id").is_in(list(valid_session_ids)))
    session_viewed_items = (
        valid_views.group_by("session_id")
        .agg(pl.col("item_id").alias("viewed_items"))
    )

    # Join
    merged = session_buy_item.join(session_viewed_items, on="session_id", how="left")

    # Check if bought item is in viewed items
    # Use Python loop for this non-trivial membership check
    n_total = len(merged)
    n_viewed = 0
    n_no_views = 0  # session had no pv events at all
    n_bought_not_viewed = 0

    for row in merged.iter_rows(named=True):
        viewed = row["viewed_items"]
        bought = row["bought_item"]
        if viewed is None:
            n_no_views += 1
        elif bought in viewed:
            n_viewed += 1
        else:
            n_bought_not_viewed += 1

    pct_viewed = n_viewed / n_total * 100
    pct_not_viewed = n_bought_not_viewed / n_total * 100
    pct_no_pv = n_no_views / n_total * 100

    print(f"\n  Valid sessions (exactly 1 buy): {n_total:,}")
    print(f"\n  Purchase was viewed (pv) before buying: {n_viewed:>8,}  ({pct_viewed:.1f}%)")
    print(f"  Purchase was NOT viewed (phantom insert): {n_bought_not_viewed:>8,}  ({pct_not_viewed:.1f}%)")
    print(f"  Session had zero pv events:               {n_no_views:>8,}  ({pct_no_pv:.1f}%)")

    if pct_not_viewed > 10:
        print(f"\n  *** WARNING: {pct_not_viewed:.1f}% of menus contain a phantom item (not observed). ***")
        print(f"      These sessions contribute artificial choices that were never part of")
        print(f"      a simultaneous consideration set.")
    elif pct_not_viewed > 5:
        print(f"\n  Note: {pct_not_viewed:.1f}% phantom inserts. Non-trivial but below critical threshold.")
    else:
        print(f"\n  Good: phantom inserts are rare ({pct_not_viewed:.1f}%). Menu construction is mostly clean.")

    return {"n_total": n_total, "n_viewed": n_viewed,
            "n_bought_not_viewed": n_bought_not_viewed, "n_no_views": n_no_views,
            "session_buy_item": session_buy_item, "session_viewed_items": session_viewed_items}


# ============================================================================
# DIAGNOSTIC 6: Purchase temporal position (assumption A3 quality)
# ============================================================================

def diagnostic_6_purchase_position(
    df_sess: pl.DataFrame,
    valid_session_ids: set,
) -> dict:
    """
    The Tenrec dataset failed because the chosen item was ALWAYS last
    (100% terminal), making the signal about sequential stopping rather
    than menu choice.

    For Taobao: what temporal position (rank) does the purchased item
    occupy among viewed items?
    - If always last → same problem as Tenrec
    - If uniformly distributed → purchase is genuinely from a set

    Also check: does viewing precede buying in time?
    (A buy event should come AFTER pv events in the same session.)
    """
    print("\n" + "="*70)
    print("D6. PURCHASE TEMPORAL POSITION (Tenrec-like chosen-last test)")
    print("="*70)

    # Use a manageable subset of valid sessions (cap at 50K for speed)
    valid_ids_list = list(valid_session_ids)[:50_000]
    subset = df_sess.filter(pl.col("session_id").is_in(valid_ids_list))

    pv_subset = subset.filter(pl.col("behavior_type") == "pv").sort(["session_id", "timestamp"])
    buy_subset = subset.filter(pl.col("behavior_type") == "buy")

    # For each session, get buy timestamp and all pv timestamps
    buy_ts = buy_subset.group_by("session_id").agg(
        pl.col("timestamp").first().alias("buy_ts"),
        pl.col("item_id").first().alias("buy_item"),
    )
    pv_ts = pv_subset.group_by("session_id").agg(
        pl.col("timestamp").alias("pv_timestamps"),
        pl.col("item_id").alias("pv_items"),
    )

    merged = buy_ts.join(pv_ts, on="session_id", how="inner")

    # Compute: what fraction of pv events come BEFORE the buy?
    # Also: is the buy item the last-viewed item?
    frac_pv_before = []
    buy_is_last_viewed = []
    buy_is_first_viewed = []
    n_pv_before_buy = []
    n_pv_after_buy = []

    for row in merged.iter_rows(named=True):
        buy_t = row["buy_ts"]
        pv_times = row["pv_timestamps"]
        pv_items = row["pv_items"]
        buy_item = row["buy_item"]

        if not pv_times:
            continue

        before = sum(1 for t in pv_times if t < buy_t)
        after = sum(1 for t in pv_times if t > buy_t)
        total = len(pv_times)

        frac_pv_before.append(before / total if total > 0 else 0)
        n_pv_before_buy.append(before)
        n_pv_after_buy.append(after)

        # Is the most recently viewed item (just before buy) the bought item?
        pv_before = [(t, i) for t, i in zip(pv_times, pv_items) if t < buy_t]
        if pv_before:
            last_viewed_item = max(pv_before, key=lambda x: x[0])[1]
            buy_is_last_viewed.append(int(last_viewed_item == buy_item))
        else:
            buy_is_last_viewed.append(0)  # no pv before buy

        if pv_before:
            first_viewed_item = min(pv_before, key=lambda x: x[0])[1]
            buy_is_first_viewed.append(int(first_viewed_item == buy_item))
        else:
            buy_is_first_viewed.append(0)

    frac_pv_before = np.array(frac_pv_before)
    n_pv_before_buy = np.array(n_pv_before_buy)
    n_pv_after_buy = np.array(n_pv_after_buy)
    buy_is_last_viewed = np.array(buy_is_last_viewed)
    buy_is_first_viewed = np.array(buy_is_first_viewed)

    print(f"\n  Sample size: {len(frac_pv_before):,} sessions")

    print(f"\n  [Key test] Fraction of pv events that occur BEFORE the buy:")
    for p in [0, 10, 25, 50, 75, 90, 100]:
        v = np.percentile(frac_pv_before, p)
        print(f"    p{p:3d}: {v*100:.1f}%")
    print(f"    Mean: {np.mean(frac_pv_before)*100:.1f}%  (100% = user only browses before buying)")

    print(f"\n  pv events before the buy (items that formed the effective menu):")
    for p in [0, 10, 25, 50, 75, 90, 95, 99]:
        v = np.percentile(n_pv_before_buy, p)
        print(f"    p{p:3d}: {v:.0f}")

    print(f"\n  pv events AFTER the buy (browsing that followed purchase):")
    pct_any_after = np.mean(n_pv_after_buy > 0) * 100
    print(f"    Sessions with any pv after buy: {pct_any_after:.1f}%")
    for p in [50, 75, 90, 95, 99]:
        v = np.percentile(n_pv_after_buy, p)
        print(f"    p{p:3d}: {v:.0f}")

    print(f"\n  [Critical] Was the bought item the LAST viewed item before purchase?")
    pct_last = np.mean(buy_is_last_viewed) * 100
    pct_first = np.mean(buy_is_first_viewed) * 100
    print(f"    Bought item = last viewed before buy: {pct_last:.1f}%")
    print(f"    Bought item = first viewed (in session): {pct_first:.1f}%")

    if pct_last > 80:
        print(f"\n  *** WARNING: Bought item is almost always the last viewed item ({pct_last:.1f}%). ***")
        print(f"      This is the same structural issue as Tenrec: the 'choice' is a stopping")
        print(f"      event, not a preference from a simultaneous set.")
    elif pct_last > 50:
        print(f"\n  Note: Bought item is often last viewed ({pct_last:.1f}%). Some recency bias.")
    else:
        print(f"\n  Good: No strong last-position bias ({pct_last:.1f}%). Choice is not mechanically terminal.")

    return {"frac_pv_before": frac_pv_before, "buy_is_last_viewed": buy_is_last_viewed,
            "n_pv_before_buy": n_pv_before_buy}


# ============================================================================
# DIAGNOSTIC 7: Menu size distribution (assumption A6)
# ============================================================================

def diagnostic_7_menu_size(
    df_sess: pl.DataFrame,
    valid_session_ids: set,
) -> dict:
    """
    A6: Menu size must be in [2, 50].

    What is the distribution of menu sizes (pv count per valid session)?
    How much data is dropped by the min=2 and max=50 filters?
    The 'menu' in the loader is the SET of viewed items - so duplicates
    (viewing the same item twice) don't expand the menu.
    """
    print("\n" + "="*70)
    print("D7. MENU SIZE DISTRIBUTION (assumption: menu size in [2, 50])")
    print("="*70)

    valid_views = (
        df_sess
        .filter(
            (pl.col("behavior_type") == "pv") &
            (pl.col("session_id").is_in(list(valid_session_ids)))
        )
    )

    # Menu = set of UNIQUE viewed items per session
    menu_sizes = (
        valid_views.group_by("session_id")
        .agg(pl.col("item_id").n_unique().alias("menu_size_unique"),
             pl.len().alias("pv_events"))  # total pv events (includes revisits)
    )

    menu_size_arr = menu_sizes["menu_size_unique"].to_numpy()
    pv_events_arr = menu_sizes["pv_events"].to_numpy()

    n_total = len(menu_size_arr)
    n_size_0 = int(np.sum(menu_size_arr == 0))  # sessions with no views at all
    n_size_1 = int(np.sum(menu_size_arr == 1))
    n_size_2_50 = int(np.sum((menu_size_arr >= 2) & (menu_size_arr <= 50)))
    n_size_gt50 = int(np.sum(menu_size_arr > 50))

    # Also count sessions from valid_session_ids with NO pv events
    all_valid = len(valid_session_ids)
    n_no_pv = all_valid - n_total  # valid sessions with zero pv events

    print(f"\n  Valid sessions (exactly 1 buy): {all_valid:,}")
    print(f"  Sessions with ≥1 pv event: {n_total:,}")
    print(f"  Sessions with zero pv events: {n_no_pv:,}  ({n_no_pv/all_valid*100:.1f}%)")

    print(f"\n  Menu size (unique viewed items) distribution:")
    print(f"    size 0 (no pv): {n_size_0+n_no_pv:>8,}  ({(n_size_0+n_no_pv)/all_valid*100:.1f}%)  DROPPED (size < 2)")
    print(f"    size 1:         {n_size_1:>8,}  ({n_size_1/all_valid*100:.1f}%)  DROPPED (size < 2, but buy inserted → size 2)")
    print(f"    size 2-50:      {n_size_2_50:>8,}  ({n_size_2_50/all_valid*100:.1f}%)  KEPT")
    print(f"    size > 50:      {n_size_gt50:>8,}  ({n_size_gt50/all_valid*100:.1f}%)  DROPPED")

    # Note: for size-1 case, the buy item is added → effective size = 2 if buy != viewed item
    # or size stays 1 if buy == viewed item
    # Let's be precise: after menu | {choice}:
    # If pv gave size=1 items and choice is the same → still size 1 → DROPPED
    # If pv gave size=1 items and choice is different → size 2 → KEPT
    # This is already handled by the loader, so just report raw pv set sizes

    print(f"\n  pv events per session (raw pageviews, including revisits):")
    for p in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
        v = np.percentile(pv_events_arr, p)
        print(f"    p{p:3d}: {v:.0f}")

    print(f"\n  Menu size (unique items) percentiles:")
    for p in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
        v = np.percentile(menu_size_arr, p)
        print(f"    p{p:3d}: {v:.0f}")

    print(f"\n  Menu size histogram [1-60]:")
    edges = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100]
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1]
        count = int(np.sum((menu_size_arr >= lo) & (menu_size_arr < hi)))
        pct = count / n_total * 100
        bar = "#" * int(pct / 0.5)
        flag = " [BOUNDARY]" if lo == 2 or lo == 50 else ""
        print(f"    [{lo:3d}-{hi:3d}): {count:>7,}  ({pct:5.1f}%)  {bar}{flag}")

    # Repeat-viewing: how often do users view the same item multiple times?
    df_merged = menu_sizes.with_columns(
        (pl.col("pv_events") - pl.col("menu_size_unique")).alias("revisits")
    )
    revisit_arr = df_merged["revisits"].to_numpy()
    pct_has_revisits = np.mean(revisit_arr > 0) * 100
    print(f"\n  Sessions where user viewed same item multiple times (revisits): {pct_has_revisits:.1f}%")
    print(f"    Mean revisit count per session (conditional on any revisit): "
          f"{np.mean(revisit_arr[revisit_arr > 0]):.1f}")

    return {"menu_size_arr": menu_size_arr, "n_size_2_50": n_size_2_50, "all_valid": all_valid}


# ============================================================================
# DIAGNOSTIC 8: Category coherence (assumption A2 quality)
# ============================================================================

def diagnostic_8_category_coherence(
    df_sess: pl.DataFrame,
    valid_session_ids: set,
) -> dict:
    """
    A menu is coherent if items within it are substitutes in the same
    product category (e.g. all shoes, all laptops).

    If a session spans many unrelated categories, the 'menu' is not a
    coherent choice set - it's a browsing trail through unrelated goods.
    A cross-category session can still be valid if it's a bundled purchase
    (choosing between product types), but RP tests implicitly assume
    the budget is spent on homogeneous goods.

    Check: how many distinct categories appear per session menu?
    What fraction of items in a menu share the majority category?
    """
    print("\n" + "="*70)
    print("D8. CATEGORY COHERENCE (do menus span multiple product categories?)")
    print("="*70)

    valid_views = (
        df_sess
        .filter(
            (pl.col("behavior_type") == "pv") &
            (pl.col("session_id").is_in(list(valid_session_ids)))
        )
    )

    # Distinct categories per session, majority-category fraction
    cat_per_session = (
        valid_views.group_by("session_id")
        .agg(
            pl.col("category_id").n_unique().alias("n_categories"),
            pl.len().alias("n_pv"),
        )
    )

    n_cats_arr = cat_per_session["n_categories"].to_numpy()
    n_sessions = len(n_cats_arr)

    print(f"\n  Sessions with pv data: {n_sessions:,}")

    print(f"\n  Distinct product categories per session menu:")
    for v in [1, 2, 3, 4, 5, 10]:
        count = int(np.sum(n_cats_arr == v)) if v < 10 else int(np.sum(n_cats_arr >= v))
        pct = count / n_sessions * 100
        op = "=" if v < 10 else ">="
        print(f"    {op}{v} categories: {count:>8,}  ({pct:.1f}%)")

    print(f"\n  n_categories percentiles:")
    for p in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
        v = np.percentile(n_cats_arr, p)
        print(f"    p{p:3d}: {v:.0f}")

    pct_single_cat = np.mean(n_cats_arr == 1) * 100
    pct_multi_cat = np.mean(n_cats_arr > 1) * 100
    print(f"\n  Single-category sessions (most coherent menus): {pct_single_cat:.1f}%")
    print(f"  Multi-category sessions (cross-category browsing): {pct_multi_cat:.1f}%")

    if pct_single_cat < 50:
        print(f"\n  *** WARNING: Only {pct_single_cat:.1f}% of menus are single-category. ***")
        print(f"      Most sessions span multiple unrelated categories.")
        print(f"      Classical RP assumes a coherent budget problem.")
    elif pct_single_cat < 70:
        print(f"\n  Note: {pct_single_cat:.1f}% single-category. Significant cross-category browsing.")
    else:
        print(f"\n  Good: {pct_single_cat:.1f}% of menus are within a single category.")

    return {"n_cats_arr": n_cats_arr, "pct_single_cat": pct_single_cat}


# ============================================================================
# DIAGNOSTIC 9: User qualification sensitivity (assumption A7)
# ============================================================================

def diagnostic_9_user_qualification(
    df_sess: pl.DataFrame,
    valid_session_ids: set,
) -> dict:
    """
    A7/A8: Users need >= 5 valid sessions (post-filters) to be included.
    The top 50K users by session count are taken.

    How many users qualify? How sensitive are results to the threshold?
    What's the distribution of valid sessions per user?
    """
    print("\n" + "="*70)
    print("D9. USER QUALIFICATION (min_sessions=5, top 50K users)")
    print("="*70)

    # Get valid sessions per user
    valid_sessions_df = df_sess.filter(
        pl.col("session_id").is_in(list(valid_session_ids))
    )

    valid_per_user = (
        valid_sessions_df.group_by("user_id")
        .agg(pl.col("session_id").n_unique().alias("n_valid_sessions"))
    )

    n_valid_per_user = valid_per_user["n_valid_sessions"].to_numpy()
    n_users_total = len(n_valid_per_user)

    print(f"\n  Total users with ≥1 valid session: {n_users_total:,}")

    print(f"\n  Valid sessions per user percentiles:")
    for p in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
        v = np.percentile(n_valid_per_user, p)
        print(f"    p{p:3d}: {v:.0f}")

    print(f"\n  Users qualifying at each min_sessions threshold:")
    for min_s in [1, 3, 5, 8, 10, 15, 20, 30, 50]:
        n_qual = int(np.sum(n_valid_per_user >= min_s))
        pct = n_qual / n_users_total * 100
        mark = " <-- LOADER DEFAULT" if min_s == 5 else ""
        print(f"    >= {min_s:2d} sessions: {n_qual:>7,} users  ({pct:.1f}%){mark}")

    # How many valid observations come from top-50K?
    valid_sorted = np.sort(n_valid_per_user)[::-1]
    top50k_sessions = int(np.sum(valid_sorted[:50000]))
    total_sessions = int(np.sum(valid_sorted))
    print(f"\n  Top 50K users (A8) cover {top50k_sessions:,} of {total_sessions:,} valid sessions")
    print(f"  ({top50k_sessions/total_sessions*100:.1f}% of all valid sessions)")

    # Active vs power users
    print(f"\n  Power-user concentration:")
    for top_n in [100, 1000, 10000, 50000]:
        top_sessions = int(np.sum(valid_sorted[:top_n]))
        pct = top_sessions / total_sessions * 100
        print(f"    Top {top_n:>6,} users: {top_sessions:>8,} sessions  ({pct:.1f}% of total)")

    return {"n_valid_per_user": n_valid_per_user, "n_users_total": n_users_total}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("TAOBAO EDA: Menu-Choice Construction Assumption Audit")
    print(f"Sample: first {SAMPLE_ROWS:,} rows of UserBehavior.csv")
    print("="*70)

    df = load_sample()

    r1 = diagnostic_1_overview(df)
    r2 = diagnostic_2_gap_distribution(df)
    r3 = diagnostic_3_session_structure(df)

    df_sess = r3["df_sess"]

    r4 = diagnostic_4_purchase_per_session(df_sess)
    valid_session_ids = r4["valid_session_ids"]

    r5 = diagnostic_5_purchase_in_menu(df_sess, valid_session_ids, r4["buys"], r4["views"])
    r6 = diagnostic_6_purchase_position(df_sess, valid_session_ids)
    r7 = diagnostic_7_menu_size(df_sess, valid_session_ids)
    r8 = diagnostic_8_category_coherence(df_sess, valid_session_ids)
    r9 = diagnostic_9_user_qualification(df_sess, valid_session_ids)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Assumption Audit Scorecard")
    print("="*70)

    print(f"""
  A1  Session gap (30 min):
      {r2['threshold_fracs'][1800]*100:.1f}% of gaps are < 30 min (stay in session)
      {(1-r2['threshold_fracs'][1800])*100:.1f}% of gaps trigger a new session

  A2  Menu = viewed items (pv):
      D8 shows {r8['pct_single_cat']:.1f}% of menus are single-category

  A3/A4  Exactly 1 buy per session:
      {r4['buys_per_session'].filter(pl.col('n_unique_buys')==1).shape[0]:,} valid sessions from {df_sess['session_id'].n_unique():,} total sessions

  A5  Buy inserted into menu if not viewed:
      {r5['n_bought_not_viewed']/r5['n_total']*100:.1f}% phantom insertions

  A6  Menu size [2, 50]:
      {r7['n_size_2_50']:,} sessions pass  ({r7['n_size_2_50']/r7['all_valid']*100:.1f}% of valid sessions)

  A7  min_sessions = 5:
      {int(np.sum(r9['n_valid_per_user'] >= 5)):,} of {r9['n_users_total']:,} users qualify  ({int(np.sum(r9['n_valid_per_user'] >= 5))/r9['n_users_total']*100:.1f}%)

  A6  Chosen-last position (Tenrec test):
      {np.mean(r6['buy_is_last_viewed'])*100:.1f}% of purchases = last viewed item before buy
    """)

    print("\nDone.")


if __name__ == "__main__":
    main()
