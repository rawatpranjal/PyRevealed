"""Open E-Commerce (Amazon) dataset loader.

Loads the Open E-Commerce 1.0 dataset of ~4,700 Amazon consumer
purchase histories, returning a BehaviorPanel.

Data must be downloaded separately.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from prefgraph.core.panel import BehaviorPanel
from prefgraph.core.session import BehaviorLog

# --- Constants ---

MIN_PRICE = 0.01
MAX_PRICE = 1000.0
TOP_N_CATEGORIES = 50
MIN_OBSERVATIONS = 5

# Category mapping: keyword -> group name
CATEGORY_GROUPS = {
    "book": "Books", "kindle": "Books",
    "electronic": "Electronics", "computer": "Electronics", "phone": "Electronics",
    "clothing": "Clothing", "apparel": "Clothing", "shoe": "Clothing",
    "home": "Home & Garden", "garden": "Home & Garden", "kitchen": "Home & Garden",
    "grocery": "Grocery", "food": "Grocery", "gourmet": "Grocery",
    "health": "Health & Beauty", "beauty": "Health & Beauty", "personal care": "Health & Beauty",
    "toy": "Toys & Games", "game": "Toys & Games",
    "sport": "Sports & Outdoors", "outdoor": "Sports & Outdoors",
    "baby": "Baby Products",
    "pet": "Pet Supplies",
    "office": "Office Products",
    "automotive": "Automotive",
    "tool": "Tools & Home Improvement",
    "music": "Music & Entertainment", "movie": "Music & Entertainment", "video": "Music & Entertainment",
}


def _map_category(category: str) -> str:
    """Map raw Amazon category to group."""
    cat_lower = str(category).lower()
    for keyword, group in CATEGORY_GROUPS.items():
        if keyword in cat_lower:
            return group
    return "Other"


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find open ecommerce data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "open_ecommerce")

    candidates.extend([
        Path.home() / ".prefgraph" / "data" / "open_ecommerce",
        Path(__file__).resolve().parents[3] / "datasets" / "open_ecommerce" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (d / "amazon-purchases.csv").exists():
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Open E-Commerce data not found. Searched:\n  {searched}\n\n"
        "Download the amazon-purchases.csv file and place it in one of the above directories."
    )


def load_open_ecommerce(
    data_dir: str | Path | None = None,
    n_users: int | None = None,
    min_observations: int = MIN_OBSERVATIONS,
    top_n_categories: int = TOP_N_CATEGORIES,
) -> BehaviorPanel:
    """Load Open E-Commerce (Amazon) dataset as a BehaviorPanel.

    Args:
        data_dir: Path to directory containing amazon-purchases.csv.
        n_users: Max number of users to include (None = all).
        min_observations: Minimum active months per user (default 5).
        top_n_categories: Number of top categories to include (default 50).

    Returns:
        BehaviorPanel with one BehaviorLog per user.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for dataset loaders. "
            "Install with: pip install 'prefgraph[datasets]'"
        ) from None

    data_path = _find_data_dir(data_dir)
    df = pd.read_csv(data_path / "amazon-purchases.csv", low_memory=False)

    # Parse dates and create monthly periods
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date"])
    df["period"] = df["Order Date"].dt.to_period("M").astype(str)

    # Map categories
    df["category"] = df["Category"].apply(_map_category)

    # Filter
    df = df[
        (df["Purchase Price Per Unit"].between(MIN_PRICE, MAX_PRICE)) &
        (df["Quantity"] > 0)
    ]

    # Select top categories by count
    top_cats = df["category"].value_counts().head(top_n_categories).index.tolist()
    df = df[df["category"].isin(top_cats)]
    categories = sorted(top_cats)

    # Build price oracle
    periods = sorted(df["period"].unique())
    price_pivot = df.pivot_table(
        values="Purchase Price Per Unit", index="period", columns="category",
        aggfunc="median",
    ).reindex(index=periods, columns=categories)
    price_pivot = price_pivot.ffill().bfill().fillna(price_pivot.median())
    price_grid = price_pivot.values  # (n_periods, n_categories)
    period_to_idx = {p: i for i, p in enumerate(periods)}

    # Build per-user sessions
    logs: dict[str, BehaviorLog] = {}
    user_col = "Survey ResponseID"
    grouped = df.groupby(user_col)
    user_ids = list(grouped.groups.keys())

    if n_users is not None:
        user_ids = user_ids[:n_users]

    for uid_raw in user_ids:
        user_data = grouped.get_group(uid_raw)

        qty_pivot = user_data.pivot_table(
            values="Quantity", index="period", columns="category",
            aggfunc="sum",
        ).reindex(columns=categories).fillna(0)

        active_periods = qty_pivot[qty_pivot.sum(axis=1) > 0].index.tolist()
        if len(active_periods) < min_observations:
            continue

        qty_matrix = qty_pivot.loc[active_periods].values
        price_indices = [period_to_idx[p] for p in active_periods if p in period_to_idx]
        if len(price_indices) != len(active_periods):
            continue
        # User-specific realized prices where available; market median otherwise
        user_price_pivot = (
            user_data.pivot_table(
                values="Purchase Price Per Unit",
                index="period",
                columns="category",
                aggfunc="median",
            )
            .reindex(columns=categories)
        )
        user_price_matrix = (
            user_price_pivot.loc[active_periods].values.astype(np.float64)
            if set(active_periods).issubset(set(user_price_pivot.index))
            else np.full((len(active_periods), len(categories)), np.nan, dtype=np.float64)
        )

        market_slice = price_grid[price_indices].astype(np.float64)
        # Fill NaNs (unbought categories) with market medians
        price_matrix = np.where(np.isnan(user_price_matrix), market_slice, user_price_matrix)

        uid = f"user_{uid_raw}"
        logs[uid] = BehaviorLog(
            cost_vectors=price_matrix,
            action_vectors=qty_matrix,
            user_id=uid,
        )

    return BehaviorPanel(
        _logs=logs,
        metadata={
            "dataset": "open_ecommerce",
            "goods": categories,
            "min_observations": min_observations,
        },
    )
