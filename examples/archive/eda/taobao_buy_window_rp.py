#!/usr/bin/env python3
"""Taobao buy-anchored window EDA + RP features.

Build menu-choice observations anchored on each buy:
  - Menu = items viewed in the last W seconds before the buy
  - Require: bought item was viewed in window
  - Filter menu size to [min_menu_size, max_menu_size]
  - Group per user and keep users with >= min_sessions

Then compute RP features (SARP/WARP/HM) via Engine.analyze_menus
and print summary statistics for comparison across window sizes.

Usage:
  python examples/eda/taobao_buy_window_rp.py --window 21600 --rows 2000000
  python examples/eda/taobao_buy_window_rp.py --window 86400 --rows 2000000

Data path defaults to ~/.prefgraph/data/taobao/UserBehavior.csv
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl


def load_sample(path: Path, n_rows: int) -> pl.DataFrame:
    print(f"  Loading {n_rows:,} rows from {path}")
    cols = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=cols,
        n_rows=n_rows,
        schema_overrides={
            "user_id": pl.Int64,
            "item_id": pl.Int64,
            "category_id": pl.Int64,
            "behavior_type": pl.Utf8,
            "timestamp": pl.Int64,
        },
    )
    print(f"  Shape: {df.shape}")
    df = df.filter(pl.col("behavior_type").is_in(["pv", "buy"]))
    return df.select(["user_id", "item_id", "behavior_type", "timestamp"]).sort(["user_id", "timestamp"])  # noqa: E501


def build_buy_window_logs(
    df: pl.DataFrame,
    window_seconds: int,
    min_menu_size: int = 2,
    max_menu_size: int = 50,
    min_sessions: int = 5,
    max_users: int | None = None,
):
    """Construct per-user menu-choice logs anchored on buy windows."""
    user = df["user_id"].to_numpy()
    item = df["item_id"].to_numpy()
    btype = df["behavior_type"].to_numpy()
    ts = df["timestamp"].to_numpy()

    # Enumerate per-user segments (start, end)
    segs: list[tuple[int, int, int]] = []
    s = 0
    for i in range(1, len(user)):
        if user[i] != user[i - 1]:
            segs.append((int(user[i - 1]), s, i))
            s = i
    segs.append((int(user[-1]), s, len(user)))

    # Per-user: single pass with sliding window over views
    events_per_user: dict[int, list[tuple[frozenset[int], int]]] = defaultdict(list)

    for uid, s, e in segs:
        u_ts = ts[s:e]
        u_item = item[s:e]
        u_type = btype[s:e]

        left = 0
        view_counts: Counter[int] = Counter()
        for j in range(e - s):
            t = u_ts[j]
            # Slide left bound to maintain [t - W, t) window
            while left < j and u_ts[left] < t - window_seconds:
                if u_type[left] == "pv":
                    it = int(u_item[left])
                    view_counts[it] -= 1
                    if view_counts[it] <= 0:
                        del view_counts[it]
                left += 1

            if u_type[j] == "buy":
                choice = int(u_item[j])
                menu_items = set(view_counts.keys())
                # Enforce: choice must have been viewed in the window
                if choice not in menu_items:
                    continue
                if not (min_menu_size <= len(menu_items) <= max_menu_size):
                    continue
                events_per_user[uid].append((frozenset(menu_items), choice))
            else:
                # Add view to current window state
                it = int(u_item[j])
                view_counts[it] += 1

    # Filter users by min_sessions and cap max_users
    user_ids = [uid for uid, ev in events_per_user.items() if len(ev) >= min_sessions]
    # Sort by number of events desc
    user_ids.sort(key=lambda u: len(events_per_user[u]), reverse=True)
    if max_users is not None:
        user_ids = user_ids[:max_users]

    # Package for Engine.analyze_menus: (menus, choices, n_items)
    tuples: list[tuple[list[list[int]], list[int], int]] = []
    for uid in user_ids:
        ev = events_per_user[uid]
        # Remap items per-user to 0..N-1
        all_items = set()
        for m, _ in ev:
            all_items |= set(m)
        item_map = {it: k for k, it in enumerate(sorted(all_items))}
        menus = [sorted(item_map[i] for i in m) for m, _ in ev]
        choices = [item_map[c] for _, c in ev]
        tuples.append((menus, choices, len(item_map)))

    return tuples, {uid: len(events_per_user[uid]) for uid in user_ids}


def summarize_menu_results(results, sessions_per_user: dict[int, int]) -> None:
    from statistics import mean
    n = len(results)
    if n == 0:
        print("  No users qualified under these settings.")
        return
    sarp_pass = sum(1 for r in results if r.is_sarp)
    warp_pass = sum(1 for r in results if r.is_warp)
    hm_ratios = [r.hm_consistent / r.hm_total if r.hm_total else 0.0 for r in results]
    n_items_max = max(r.max_scc for r in results)
    print(f"  Users: {n:,}")
    print(f"  SARP pass: {sarp_pass/n:.1%}  |  WARP pass: {warp_pass/n:.1%}")
    print(f"  HM ratio (mean): {mean(hm_ratios):.3f}")
    # Sessions per user quick summary
    sp = list(sessions_per_user.values())
    print(f"  Sessions per user: mean={mean(sp):.1f}  p50={np.percentile(sp,50):.0f}  p90={np.percentile(sp,90):.0f}")


def main():
    parser = argparse.ArgumentParser(description="Taobao buy-anchored window RP EDA")
    parser.add_argument("--data", type=str, default=str(Path.home() / ".prefgraph" / "data" / "taobao" / "UserBehavior.csv"))
    parser.add_argument("--rows", type=int, default=2_000_000, help="Number of CSV rows to load")
    parser.add_argument("--window", type=int, default=1800, help="Window size in seconds (e.g. 1800=30m, 21600=6h, 86400=1d)")
    parser.add_argument("--min-sessions", type=int, default=5)
    parser.add_argument("--max-users", type=int, default=2000, help="Cap users for RP to keep runtime modest")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"Data not found: {path}")
        return

    df = load_sample(path, args.rows)
    tuples, sessions_per_user = build_buy_window_logs(
        df,
        window_seconds=args.window,
        min_sessions=args.min_sessions,
        max_users=args.max_users,
    )
    print(f"\nBuy-anchored window={args.window//3600}h  |  qualifying users: {len(tuples):,}")

    # Compute RP features
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from prefgraph.engine import Engine

    engine = Engine()
    results = engine.analyze_menus(tuples)

    # Summarize
    summarize_menu_results(results, sessions_per_user)


if __name__ == "__main__":
    main()

