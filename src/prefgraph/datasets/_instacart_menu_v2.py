"""Instacart V2 menu-choice loader: aisle-level single-reorder with trailing-3 menus.

Construction spec:
  Observation = user × order × aisle
  Choice      = sole reordered SKU in that (user, order, aisle) triple
  Menu        = products bought in trailing-3 orders in same aisle, union {choice}
  Filters:
    - menu_size >= min_menu_size  (default 2)
    - (user, aisle) pair count >= min_pair_events  (default 3)
    - user total events >= min_sessions  (default 5)

This replaces the old department + first-in-cart construction.

Replaces: _instacart_menu.py (V1, deprecated for benchmarking)
Data: ~/.prefgraph/data/instacart/ (Kaggle Market Basket Analysis)
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import polars as pl

from prefgraph.core.session import MenuChoiceLog


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Search standard locations for the Instacart data directory."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    env = os.environ.get("PYREVEALED_DATA_DIR") or os.environ.get("PREFGRAPH_DATA_DIR")
    if env:
        candidates.append(Path(env) / "instacart")
    candidates.extend([
        Path.home() / ".prefgraph" / "data" / "instacart",
        Path.home() / ".pyrevealed" / "data" / "instacart",
    ])
    for d in candidates:
        if d.is_dir() and (d / "orders.csv").exists():
            return d
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Instacart data not found. Searched:\n  {searched}")


def load_instacart_menu_v2(
    data_dir: str | Path | None = None,
    max_users: int | None = 50_000,
    min_sessions: int = 5,
    min_menu_size: int = 2,
    min_pair_events: int = 3,
) -> dict[str, MenuChoiceLog]:
    """Load Instacart as aisle-level menu choices with trailing-3 familiar menus.

    Each observation is a (user, order, aisle) triple where:

    - The user reordered exactly one SKU in that aisle (choice)
    - The menu is all distinct products the user bought in that aisle
      across their previous 3 orders, plus the current choice

    Invariants on returned logs:
    - Every choice is in its menu
    - Every menu has at least min_menu_size items
    - Every user has at least min_sessions observations

    Args:
        data_dir: Path to Instacart data directory. None uses standard locations.
        max_users: Cap on users returned. None returns all.
        min_sessions: Minimum valid observations per user (default 5).
        min_menu_size: Minimum menu cardinality (default 2).
        min_pair_events: Minimum events per (user, aisle) pair (default 3).

    Returns:
        Dict mapping "user_{id}" -> MenuChoiceLog, sorted by order_number.
    """
    data_path = _find_data_dir(data_dir)
    print(f"  Loading Instacart V2 data from {data_path}...")

    # --- Step 1: Load raw tables ---
    # Use only prior orders (historical behavior, not the held-out train order)
    orders = (
        pl.read_csv(data_path / "orders.csv")
        .filter(pl.col("eval_set") == "prior")
        .select(["order_id", "user_id", "order_number"])
    )

    # Read product-level order data (not add_to_cart_order - intentionally excluded)
    op = pl.read_csv(
        data_path / "order_products__prior.csv",
        columns=["order_id", "product_id", "reordered"],
    )

    # Map products to aisles
    products = pl.read_csv(
        data_path / "products.csv",
        columns=["product_id", "aisle_id"],
    )

    # --- Step 2: Build full (user, order_number, aisle_id, product_id, reordered) ---
    df = (
        op
        .join(products, on="product_id")
        .join(orders, on="order_id")
        .select(["user_id", "order_number", "aisle_id", "product_id", "reordered"])
        .sort(["user_id", "order_number", "aisle_id"])
    )

    print(f"  {len(df):,} product rows, {df['user_id'].n_unique():,} users")

    # --- Step 3: Identify single-reorder events per (user, order, aisle) ---
    # A valid event has exactly 1 reordered SKU in that aisle-order combination.
    # ~80% of aisle-order combos have exactly 1 reordered product.
    single_reorder = (
        df
        .filter(pl.col("reordered") == 1)
        .group_by(["user_id", "order_number", "aisle_id"])
        .agg([
            pl.len().alias("n_reordered"),
            pl.col("product_id").first().alias("choice_product_id"),
        ])
        .filter(pl.col("n_reordered") == 1)
        .drop("n_reordered")
        .sort(["user_id", "order_number", "aisle_id"])
    )

    print(f"  {len(single_reorder):,} single-reorder (user, order, aisle) events")

    # --- Step 4: Build aisle history for trailing-3 menu construction ---
    # For each (user, aisle, order_number), collect ALL products bought (not just reorders).
    # This forms the "familiarity set" the trailing-3 window draws from.
    aisle_history = (
        df
        .group_by(["user_id", "aisle_id", "order_number"])
        .agg(pl.col("product_id").unique().alias("products"))
        .sort(["user_id", "aisle_id", "order_number"])
    )

    # Build lookup dict: user_id -> aisle_id -> [(order_number, frozenset(products))]
    # Lists are in ascending order_number order (from the sort above).
    history_dict: dict[int, dict[int, list[tuple[int, frozenset]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in aisle_history.iter_rows(named=True):
        history_dict[row["user_id"]][row["aisle_id"]].append(
            (row["order_number"], frozenset(row["products"]))
        )

    # --- Step 5: Construct events with trailing-3 menus ---
    # For each single-reorder event, look at the user's previous 3 orders in the
    # same aisle, collect all distinct products from those orders, and add the
    # current choice to form the menu. This trailing-3 window is tighter than
    # all-time history and avoids inflating menus with stale items.
    built_events: list[dict] = []

    for row in single_reorder.iter_rows(named=True):
        uid = row["user_id"]
        aid = row["aisle_id"]
        onum = row["order_number"]
        choice = row["choice_product_id"]

        # Trailing-3: all products bought in this aisle in the 3 orders before current
        ua_hist = history_dict[uid][aid]
        prev_orders = [entry for entry in ua_hist if entry[0] < onum]
        trailing_3 = prev_orders[-3:]  # last 3 orders before current

        menu_products: set[int] = set()
        for _, prods in trailing_3:
            menu_products |= prods

        # Guarantee: choice is always in menu (invariant)
        menu_products.add(choice)

        # Filter: menu_size >= min_menu_size
        if len(menu_products) < min_menu_size:
            continue

        built_events.append({
            "user_id": uid,
            "aisle_id": aid,
            "order_number": onum,
            "choice_product_id": choice,
            "menu_product_ids": sorted(menu_products),
            "menu_size": len(menu_products),
        })

    print(f"  {len(built_events):,} events after menu_size >= {min_menu_size} filter")

    # --- Step 6: Filter (user, aisle) pairs with >= min_pair_events ---
    # This removes pairs with too few observations for meaningful RP analysis.
    # After this filter, median events per pair rises to ~5.
    events_df = pl.DataFrame(
        built_events,
        schema={
            "user_id": pl.Int64,
            "aisle_id": pl.Int64,
            "order_number": pl.Int64,
            "choice_product_id": pl.Int64,
            "menu_product_ids": pl.List(pl.Int64),
            "menu_size": pl.Int64,
        },
    )

    pair_counts = (
        events_df
        .group_by(["user_id", "aisle_id"])
        .agg(pl.len().alias("pair_count"))
        .filter(pl.col("pair_count") >= min_pair_events)
    )

    events_df = events_df.join(
        pair_counts.select(["user_id", "aisle_id"]),
        on=["user_id", "aisle_id"],
    ).sort(["user_id", "order_number", "aisle_id"])

    n_pairs = events_df.select(["user_id", "aisle_id"]).unique().height
    print(
        f"  {len(events_df):,} events, "
        f"{events_df['user_id'].n_unique():,} users, "
        f"{n_pairs:,} user-aisle pairs "
        f"(after pair_count >= {min_pair_events} filter)"
    )

    # --- Step 7: Package into per-user MenuChoiceLog ---
    # Each user gets one MenuChoiceLog covering all their qualifying user-aisle
    # pairs, sorted by order_number. Item IDs are remapped to compact 0..N-1.
    user_logs: dict[str, MenuChoiceLog] = {}
    n_qualified = 0

    for (uid,), group in events_df.group_by(["user_id"], maintain_order=True):
        group = group.sort("order_number")
        rows = list(group.iter_rows(named=True))

        menus_raw: list[frozenset] = []
        choices_raw: list[int] = []
        for r in rows:
            menus_raw.append(frozenset(r["menu_product_ids"]))
            choices_raw.append(r["choice_product_id"])

        if len(choices_raw) < min_sessions:
            continue

        # Compact item remapping: original product_ids -> 0..N-1
        all_items: set[int] = set()
        for m in menus_raw:
            all_items |= m
        item_map = {item: idx for idx, item in enumerate(sorted(all_items))}

        menus = [frozenset(item_map[i] for i in m) for m in menus_raw]
        choices = [item_map[c] for c in choices_raw]

        user_logs[f"user_{uid}"] = MenuChoiceLog(menus=menus, choices=choices)
        n_qualified += 1

        if max_users is not None and n_qualified >= max_users:
            break

    print(f"  {len(user_logs):,} users with >= {min_sessions} sessions")
    return user_logs
