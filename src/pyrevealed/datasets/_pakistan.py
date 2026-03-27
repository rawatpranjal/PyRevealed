"""Pakistan Largest E-Commerce dataset loader.

Loads the Pakistan e-commerce dataset (~1M+ transactions across 16 product
categories) with real prices and quantities, returning a BehaviorPanel.

Budget-based: each observation is a customer-month with price and quantity
vectors across 16 product categories.

Data must be downloaded separately from Kaggle:
https://www.kaggle.com/datasets/zusmani/pakistans-largest-ecommerce-dataset
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pyrevealed.core.panel import BehaviorPanel
from pyrevealed.core.session import BehaviorLog

# --- Constants ---

CATEGORIES = [
    "Mobiles & Tablets", "Entertainment", "Computing", "Appliances",
    "Men's Fashion", "Women's Fashion", "Kids & Baby",
    "Superstore", "Beauty & Grooming", "Health & Sports",
    "Home & Living", "Books", "School & Education",
    "Soghaat", "Others",
]

CSV_FILENAME = "Pakistan Largest Ecommerce Dataset.csv"

NUM_CATEGORIES = len(CATEGORIES)
MIN_UNIT_PRICE = 0.01
MAX_UNIT_PRICE = 500_000.0  # PKR — electronics can be expensive


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find Pakistan e-commerce data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "pakistan")

    candidates.extend([
        Path.home() / ".pyrevealed" / "data" / "pakistan",
        Path(__file__).resolve().parents[3] / "pakistan" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (d / CSV_FILENAME).exists():
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Pakistan e-commerce data not found. Searched:\n  {searched}\n\n"
        "Download from Kaggle: "
        "https://www.kaggle.com/datasets/zusmani/pakistans-largest-ecommerce-dataset\n"
        "Place the CSV in ~/.pyrevealed/data/pakistan/ or pass data_dir=."
    )


def load_pakistan(
    data_dir: str | Path | None = None,
    max_users: int = 50_000,
    min_months: int = 5,
) -> BehaviorPanel:
    """Load Pakistan E-Commerce dataset as a BehaviorPanel.

    Filters to completed orders with positive price/quantity and non-null
    Customer ID. Aggregates to monthly periods: for each customer-month,
    quantity = total units ordered per category, price = median unit price
    per category that month (market-wide oracle).

    All 16 product categories are used as goods.

    Args:
        data_dir: Path to directory containing the CSV.
            If None, searches standard locations.
        max_users: Maximum number of customers to include (default 50,000).
        min_months: Minimum active months per customer (default 5).

    Returns:
        BehaviorPanel with one BehaviorLog per customer (rows = months,
        cols = product categories).

    Raises:
        FileNotFoundError: If data file cannot be found.
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

    # --- Load CSV ---
    df = pd.read_csv(
        data_path / CSV_FILENAME,
        usecols=[
            "item_id", "status", "created_at", "price", "qty_ordered",
            "grand_total", "category_name_1", "payment_method",
            "Customer ID", "Year", "Month",
        ],
        dtype={"Customer ID": str},
    )

    # --- Filter ---
    # Only completed orders
    df = df[df["status"].str.strip().str.lower() == "complete"].copy()

    # Non-null Customer ID
    df = df.dropna(subset=["Customer ID"])
    df = df[df["Customer ID"].str.strip() != ""]

    # Positive price and quantity
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty_ordered"] = pd.to_numeric(df["qty_ordered"], errors="coerce")
    df = df.dropna(subset=["price", "qty_ordered"])
    df = df[(df["price"] > 0) & (df["qty_ordered"] > 0)]

    # Price sanity bounds
    df = df[
        (df["price"] >= MIN_UNIT_PRICE) &
        (df["price"] <= MAX_UNIT_PRICE)
    ]

    # Non-null category
    df = df.dropna(subset=["category_name_1"])
    df["category"] = df["category_name_1"].str.strip()

    # --- Discover categories from data ---
    # Use hardcoded list as preference order, but accept whatever exists
    data_categories = df["category"].value_counts().index.tolist()
    categories = [c for c in CATEGORIES if c in data_categories]
    # Add any categories present in data but not in our hardcoded list
    for c in data_categories:
        if c not in categories:
            categories.append(c)

    df = df[df["category"].isin(categories)]

    # --- Build year-month period key ---
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df = df.dropna(subset=["Year", "Month"])
    df["year_month"] = (
        df["Year"].astype(int).astype(str) + "-" +
        df["Month"].astype(int).astype(str).str.zfill(2)
    )

    # --- Build price oracle: median price per category per month ---
    price_oracle = df.pivot_table(
        values="price", index="year_month", columns="category",
        aggfunc="median",
    ).reindex(columns=categories)
    price_oracle = price_oracle.ffill().bfill()

    # Fill remaining NaN with global median per category
    global_medians = df.groupby("category")["price"].median()
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
    agg = df.groupby(
        ["Customer ID", "year_month", "category"], observed=True,
    ).agg(
        total_qty=("qty_ordered", "sum"),
    ).reset_index()

    # --- Build per-customer BehaviorLogs ---
    logs: dict[str, BehaviorLog] = {}

    grouped = agg.groupby("Customer ID")
    customer_ids = list(grouped.groups.keys())

    # Cap at max_users
    if max_users is not None and len(customer_ids) > max_users:
        customer_ids = customer_ids[:max_users]

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

        uid = f"customer_{cust_id}"
        logs[uid] = BehaviorLog(
            cost_vectors=price_matrix.astype(np.float64),
            action_vectors=qty_matrix.astype(np.float64),
            user_id=uid,
            metadata={
                "active_months": list(active_months),
            },
        )

        # Early exit if we hit max_users worth of valid logs
        if max_users is not None and len(logs) >= max_users:
            break

    return BehaviorPanel(
        _logs=logs,
        metadata={
            "dataset": "pakistan_ecommerce",
            "goods": categories,
            "n_categories": len(categories),
            "min_months": min_months,
            "max_users": max_users,
            "n_customers": len(logs),
            "total_months": len(all_months),
        },
    )
