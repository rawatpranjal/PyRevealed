"""Olist Brazilian E-Commerce dataset loader.

Loads the Olist dataset of ~100K orders from ~96K customers across
Brazilian marketplaces, returning a BehaviorPanel.

Key: Olist anonymizes customer_id per order. The true persistent
identifier is customer_unique_id in olist_customers_dataset.csv.
~3K customers have 2+ orders; ~250 have 3+.

Data must be downloaded separately from Kaggle:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pyrevealed.core.panel import BehaviorPanel
from pyrevealed.core.session import BehaviorLog

# --- Constants ---

TOP_CATEGORIES = [
    "bed_bath_table", "health_beauty", "sports_leisure",
    "furniture_decor", "computers_accessories", "housewares",
    "watches_gifts", "telephony", "garden_tools", "auto",
    "toys", "cool_stuff", "perfumery", "baby",
    "electronics", "stationery", "fashion_bags_accessories",
    "pet_shop", "office_furniture", "luggage_accessories",
]

NUM_CATEGORIES = len(TOP_CATEGORIES)
MIN_UNIT_PRICE = 0.01
MAX_UNIT_PRICE = 5000.0


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find Olist data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "olist")

    candidates.extend([
        Path.home() / ".pyrevealed" / "data" / "olist",
        Path(__file__).resolve().parents[3] / "olist" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (d / "olist_orders_dataset.csv").exists():
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Olist data not found. Searched:\n  {searched}\n\n"
        "Download from Kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce\n"
        "Then pass data_dir= or set PYREVEALED_DATA_DIR environment variable."
    )


def load_olist(
    data_dir: str | Path | None = None,
    n_customers: int | None = None,
    min_months: int = 2,
    min_orders: int = 3,
    n_categories: int = NUM_CATEGORIES,
) -> BehaviorPanel:
    """Load Olist Brazilian E-Commerce dataset as a BehaviorPanel.

    Joins orders, order items, products, and the customer identity table
    to build monthly budget vectors (price x quantity) across product
    categories per customer.

    Olist anonymizes customer_id per order; the true repeat-buyer key is
    customer_unique_id from olist_customers_dataset.csv. ~3K customers
    have 2+ orders, ~250 have 3+.

    Args:
        data_dir: Path to directory containing Olist CSV files.
            If None, searches standard locations.
        n_customers: Max number of customers to include (None = all).
        min_months: Minimum active months per customer (default 2).
        min_orders: Minimum number of distinct orders per customer
            (default 3). Most Olist customers have only 1 order.
        n_categories: Number of top product categories to use (default 20).

    Returns:
        BehaviorPanel with one BehaviorLog per customer (rows = months,
        cols = product categories).

    Raises:
        FileNotFoundError: If data files cannot be found.
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for dataset loaders. "
            "Install with: pip install 'pyrevealed[datasets]'"
        ) from None

    data_path = _find_data_dir(data_dir)

    # --- Load CSVs ---
    orders = pd.read_csv(
        data_path / "olist_orders_dataset.csv",
        usecols=["order_id", "customer_id", "order_status",
                 "order_purchase_timestamp"],
        parse_dates=["order_purchase_timestamp"],
    )
    customers = pd.read_csv(
        data_path / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_unique_id"],
    )
    items = pd.read_csv(
        data_path / "olist_order_items_dataset.csv",
        usecols=["order_id", "product_id", "price", "freight_value"],
    )
    products = pd.read_csv(
        data_path / "olist_products_dataset.csv",
        usecols=["product_id", "product_category_name"],
    )
    translation = pd.read_csv(
        data_path / "product_category_name_translation.csv",
        encoding="utf-8-sig",
    )

    # --- Join and filter ---
    # Only delivered orders
    orders = orders[orders["order_status"] == "delivered"].copy()

    # Resolve persistent customer identity
    orders = orders.merge(customers, on="customer_id", how="left")

    # Merge items -> products -> translation -> orders
    merged = items.merge(products, on="product_id", how="left")
    merged = merged.merge(
        translation, on="product_category_name", how="left",
    )
    merged = merged.merge(
        orders[["order_id", "customer_unique_id",
                "order_purchase_timestamp"]],
        on="order_id", how="inner",
    )

    # Use English category names, drop unmapped
    merged["category"] = merged["product_category_name_english"]
    merged = merged.dropna(subset=["category"])

    # Filter valid prices
    merged = merged[
        (merged["price"] >= MIN_UNIT_PRICE) &
        (merged["price"] <= MAX_UNIT_PRICE)
    ]

    # --- Select top categories ---
    categories = TOP_CATEGORIES[:n_categories]
    category_counts = merged["category"].value_counts()
    # Verify hardcoded list against actual data; fall back to data-driven
    available = [c for c in categories if c in category_counts.index]
    if len(available) < n_categories:
        extras = [
            c for c in category_counts.index
            if c not in available
        ]
        available.extend(extras[:n_categories - len(available)])
    categories = available[:n_categories]

    merged = merged[merged["category"].isin(categories)]

    # --- Build monthly period key ---
    merged["year_month"] = (
        merged["order_purchase_timestamp"].dt.to_period("M")
    )

    # --- Pre-filter: keep only repeat customers (by distinct orders) ---
    order_counts = (
        merged.groupby("customer_unique_id")["order_id"].nunique()
    )
    repeat_customers = order_counts[order_counts >= min_orders].index
    merged = merged[merged["customer_unique_id"].isin(repeat_customers)]

    if merged.empty:
        return BehaviorPanel(
            _logs={},
            metadata={
                "dataset": "olist",
                "goods": categories,
                "n_categories": len(categories),
                "min_months": min_months,
                "min_orders": min_orders,
                "n_customers": 0,
                "total_months": 0,
            },
        )

    # --- Build price oracle: median price per category per month ---
    # Use ALL delivered rows (before customer filter) for robust medians
    price_oracle = merged.pivot_table(
        values="price", index="year_month", columns="category",
        aggfunc="median",
    ).reindex(columns=categories)
    price_oracle = price_oracle.ffill().bfill()
    # Fill remaining NaN with global median per category
    global_medians = merged.groupby("category")["price"].median()
    for cat in categories:
        if cat in global_medians.index:
            price_oracle[cat] = price_oracle[cat].fillna(
                global_medians[cat]
            )
    price_oracle = price_oracle.fillna(1.0)  # absolute fallback

    all_months = sorted(price_oracle.index)
    month_to_idx = {m: i for i, m in enumerate(all_months)}
    price_grid = price_oracle.values  # (n_months, n_categories)

    # --- Aggregate quantity per customer-month-category ---
    # Each order item counts as quantity 1 (marketplace items)
    merged["quantity"] = 1
    agg = merged.groupby(
        ["customer_unique_id", "year_month", "category"], observed=True,
    ).agg(
        total_qty=("quantity", "sum"),
        first_timestamp=("order_purchase_timestamp", "first"),
    ).reset_index()

    # --- Build per-customer BehaviorLogs ---
    logs: dict[str, BehaviorLog] = {}

    grouped = agg.groupby("customer_unique_id")
    customer_ids = list(grouped.groups.keys())

    if n_customers is not None:
        customer_ids = customer_ids[:n_customers]

    for cust_id in customer_ids:
        cust_data = grouped.get_group(cust_id)

        # Pivot to quantity matrix (months x categories)
        qty_pivot = cust_data.pivot_table(
            values="total_qty", index="year_month", columns="category",
            aggfunc="sum",
        ).reindex(columns=categories).fillna(0)

        # Only keep months with at least one purchase
        active_months = (
            qty_pivot[qty_pivot.sum(axis=1) > 0].index.tolist()
        )
        if len(active_months) < min_months:
            continue

        qty_matrix = qty_pivot.loc[active_months].values  # (T, K)

        # Price matrix from oracle
        month_indices = [month_to_idx[m] for m in active_months]
        price_matrix = price_grid[month_indices]  # (T, K)

        # Timestamps for metadata (first purchase in each active month)
        timestamps = []
        for m in active_months:
            month_rows = cust_data[cust_data["year_month"] == m]
            ts = month_rows["first_timestamp"].min()
            timestamps.append(str(ts))

        uid = f"customer_{cust_id}"
        logs[uid] = BehaviorLog(
            cost_vectors=price_matrix.astype(np.float64),
            action_vectors=qty_matrix.astype(np.float64),
            user_id=uid,
            metadata={
                "order_purchase_timestamps": timestamps,
                "active_months": [str(m) for m in active_months],
            },
        )

    return BehaviorPanel(
        _logs=logs,
        metadata={
            "dataset": "olist",
            "goods": categories,
            "n_categories": len(categories),
            "min_months": min_months,
            "min_orders": min_orders,
            "n_customers": len(logs),
            "total_months": len(all_months),
        },
    )
