"""Instacart menu-choice loader: product-level within departments.

Constructs menu-choice observations from Instacart grocery data:
- Menu = products the user has previously bought from a department (known set)
- Choice = product they bought this order (first pick by add_to_cart_order)
- Availability confirmed by other users purchasing the same products

Each user's most active department is used. Observations are ordered
chronologically by order_number.

Data: ~/.prefgraph/data/instacart/ (Kaggle Market Basket Analysis)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from prefgraph.core.session import MenuChoiceLog


def _find_data_dir(data_dir: str | Path | None) -> Path:
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


def load_instacart_menu(
    data_dir: str | Path | None = None,
    max_users: int | None = 50_000,
    min_sessions: int = 5,
    min_menu_size: int = 3,
    max_menu_size: int = 30,
) -> dict[str, MenuChoiceLog]:
    """Load Instacart as department-scoped product-level menu choices.

    For each user, picks their most active department and constructs
    menu-choice observations where:
    - Menu = products previously bought from that department (growing known set)
    - Choice = product bought this order (first by add_to_cart_order)

    Only reorder events count (choice must be in known set).

    Args:
        data_dir: Path to Instacart data directory.
        max_users: Cap on users returned.
        min_sessions: Minimum valid menu-choice observations per user.
        min_menu_size: Minimum products in menu.
        max_menu_size: Maximum products in menu.

    Returns:
        Dict mapping user_id (str) -> MenuChoiceLog.
    """
    data_path = _find_data_dir(data_dir)
    print(f"  Loading Instacart menu-choice data from {data_path}...")

    orders = pd.read_csv(data_path / "orders.csv")
    op = pd.read_csv(data_path / "order_products__prior.csv")
    products = pd.read_csv(data_path / "products.csv")

    # Join
    op = op.merge(products[["product_id", "department_id"]], on="product_id")
    op = op.merge(orders[["order_id", "user_id", "order_number"]], on="order_id")
    op = op.sort_values(["user_id", "order_number", "add_to_cart_order"])

    print(f"  {len(op):,} order-product rows, {op['user_id'].nunique():,} users")

    # For each user, find their most active department (most reorder events)
    reorders = op[op["reordered"] == 1]
    user_dept_counts = reorders.groupby(["user_id", "department_id"]).size().reset_index(name="count")
    user_best_dept = user_dept_counts.sort_values("count", ascending=False).drop_duplicates("user_id")
    user_dept_map = dict(zip(user_best_dept["user_id"], user_best_dept["department_id"]))

    # Build menu-choice sequences
    user_logs: dict[str, MenuChoiceLog] = {}
    n_qualified = 0

    for uid, best_dept in user_dept_map.items():
        # Get this user's orders in their best department
        user_dept = op[(op["user_id"] == uid) & (op["department_id"] == best_dept)]
        if len(user_dept) == 0:
            continue

        known_products: set[int] = set()
        menus: list[frozenset] = []
        choices: list[int] = []

        for order_num, group in user_dept.sort_values("order_number").groupby("order_number"):
            products_this_order = set(group["product_id"].tolist())
            # First pick = product with lowest add_to_cart_order
            first_pick = group.iloc[0]["product_id"]

            # Valid observation: known set is big enough AND first pick is a reorder
            if (len(known_products) >= min_menu_size
                    and len(known_products) <= max_menu_size
                    and first_pick in known_products):
                menus.append(frozenset(known_products))
                choices.append(first_pick)

            # Update known set with ALL products from this order
            known_products |= products_this_order

        if len(menus) < min_sessions:
            continue

        # Remap to 0..N-1
        all_items: set[int] = set()
        for m in menus:
            all_items |= m
        item_map = {item: idx for idx, item in enumerate(sorted(all_items))}
        menus = [frozenset(item_map[i] for i in m) for m in menus]
        choices = [item_map[c] for c in choices]

        user_logs[f"user_{uid}"] = MenuChoiceLog(menus=menus, choices=choices)
        n_qualified += 1

        if max_users is not None and n_qualified >= max_users:
            break

    print(f"  Users with >= {min_sessions} valid observations: {len(user_logs):,}")
    print(f"  Built {len(user_logs)} MenuChoiceLog objects")
    return user_logs
