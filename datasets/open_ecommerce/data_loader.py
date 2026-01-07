"""Data loading and preprocessing for Open E-Commerce 1.0 dataset.

This module handles:
1. Loading the amazon-purchases.csv file
2. Parsing dates and creating time periods (monthly)
3. Extracting and grouping product categories
4. Computing price oracle (median prices per period)

Data columns (from Scientific Data paper):
- id: User ID
- order_date: Purchase date
- category: Amazon product category
- title: Product title
- price: Price in USD
- quantity: Number of items
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CACHE_DIR,
    CATEGORY_GROUPS,
    DATA_FILE,
    MAX_PRICE,
    MIN_OBSERVATIONS,
    MIN_PRICE,
    TOP_N_CATEGORIES,
)


def load_raw_data() -> pd.DataFrame:
    """
    Load raw purchase data from CSV file.

    Returns:
        Raw purchases DataFrame

    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_FILE}\n"
            "Please run download.py first to download the dataset."
        )

    print(f"  Loading from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  Raw purchases: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    return df


def map_category(category: str) -> str:
    """
    Map Amazon category to simplified category group.

    Args:
        category: Original Amazon category

    Returns:
        Simplified category name
    """
    if pd.isna(category):
        return "Other"

    category_lower = str(category).lower()

    for group_name, keywords in CATEGORY_GROUPS.items():
        for keyword in keywords:
            if keyword.lower() in category_lower:
                return group_name

    return "Other"


def filter_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and preprocess purchase data.

    Steps:
    1. Parse dates and create monthly periods
    2. Map categories to simplified groups
    3. Filter invalid prices and quantities
    4. Select top N categories by volume

    Args:
        df: Raw purchase DataFrame

    Returns:
        Filtered DataFrame with columns:
        - user_id: User identifier
        - period: Year-month string (e.g., "2020-01")
        - category: Simplified category name
        - quantity: Units purchased
        - unit_price: Price per unit (USD)
    """
    df = df.copy()

    # Identify columns (handle different column name conventions)
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "responseid" in col_lower or "survey" in col_lower:
            col_map["user_id"] = col  # Survey ResponseID is the user identifier
        elif "date" in col_lower:
            col_map["date"] = col
        elif "category" in col_lower:
            col_map["category"] = col
        elif "price" in col_lower:
            col_map["price"] = col
        elif "quantity" in col_lower or "qty" in col_lower:
            col_map["quantity"] = col

    print(f"  Column mapping: {col_map}")

    # Extract required columns
    if "user_id" not in col_map:
        # Fallback to Survey ResponseID explicitly
        if "Survey ResponseID" in df.columns:
            col_map["user_id"] = "Survey ResponseID"
        else:
            col_map["user_id"] = df.columns[0]

    user_col = col_map.get("user_id", "Survey ResponseID")
    date_col = col_map.get("date", "order_date")
    category_col = col_map.get("category", "category")
    price_col = col_map.get("price", "price")
    quantity_col = col_map.get("quantity", "quantity")

    # 1. Parse dates
    df["date_parsed"] = pd.to_datetime(df[date_col], errors="coerce")
    n_invalid_date = df["date_parsed"].isna().sum()
    df = df[df["date_parsed"].notna()]
    print(f"  Removed {n_invalid_date:,} rows with invalid dates")

    df["period"] = df["date_parsed"].dt.to_period("M").astype(str)

    # 2. Use raw categories (no grouping for better granularity)
    if category_col in df.columns:
        df["category_group"] = df[category_col].fillna("Other").astype(str)
    else:
        df["category_group"] = "Other"

    # 3. Handle quantity (default to 1 if missing)
    if quantity_col in df.columns:
        df["qty"] = pd.to_numeric(df[quantity_col], errors="coerce").fillna(1)
    else:
        df["qty"] = 1

    # Filter invalid quantities
    valid_qty = df["qty"] > 0
    n_invalid_qty = (~valid_qty).sum()
    df = df[valid_qty]
    print(f"  Removed {n_invalid_qty:,} rows with non-positive quantity")

    # 4. Handle price
    df["unit_price"] = pd.to_numeric(df[price_col], errors="coerce")

    # Filter invalid prices
    valid_price = (
        (df["unit_price"] >= MIN_PRICE)
        & (df["unit_price"] <= MAX_PRICE)
        & df["unit_price"].notna()
    )
    n_invalid_price = (~valid_price).sum()
    df = df[valid_price]
    print(f"  Removed {n_invalid_price:,} rows with invalid prices")

    # 5. Select top N categories by transaction count
    category_counts = df["category_group"].value_counts()
    top_categories = category_counts.head(TOP_N_CATEGORIES).index.tolist()

    # Always include "Other" if it exists
    if "Other" in category_counts.index and "Other" not in top_categories:
        top_categories.append("Other")

    df = df[df["category_group"].isin(top_categories)]
    print(f"  Filtered to top {len(top_categories)} categories: {len(df):,} transactions")

    # 6. Create result DataFrame
    result = pd.DataFrame({
        "user_id": df[user_col],
        "period": df["period"],
        "category": df["category_group"],
        "quantity": df["qty"].astype(float),
        "unit_price": df["unit_price"].astype(float),
    })

    print(f"  Final transactions: {len(result):,}")
    print(f"  Unique users: {result['user_id'].nunique():,}")
    print(f"  Unique periods: {result['period'].nunique()}")
    print(f"  Unique categories: {result['category'].nunique()}")

    return result


def build_price_oracle(
    filtered_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, list[str], list[str]]:
    """
    Build the master price grid (periods x categories).

    For each period and category, compute the median price across all users.
    Forward/backward fill any gaps.

    Args:
        filtered_data: Filtered transaction DataFrame

    Returns:
        Tuple of:
        - price_grid: DataFrame with periods as index, categories as columns
        - periods: List of period strings (sorted)
        - categories: List of category names (sorted)
    """
    periods = sorted(filtered_data["period"].unique())
    categories = sorted(filtered_data["category"].unique())

    print(f"  Building price grid: {len(periods)} periods x {len(categories)} categories")

    # Compute median price per period per category
    price_pivot = filtered_data.pivot_table(
        index="period",
        columns="category",
        values="unit_price",
        aggfunc="median",
    )

    # Reindex to ensure all periods and categories are present
    price_pivot = price_pivot.reindex(index=periods, columns=categories)

    # Forward fill then backward fill
    price_pivot = price_pivot.ffill().bfill()

    # Fill any remaining NaN with global median
    n_nan = price_pivot.isna().sum().sum()
    if n_nan > 0:
        print(f"  Warning: {n_nan} missing prices after fill, using global median")
        for col in price_pivot.columns:
            if price_pivot[col].isna().any():
                global_median = filtered_data.loc[
                    filtered_data["category"] == col, "unit_price"
                ].median()
                price_pivot[col] = price_pivot[col].fillna(global_median)

    return price_pivot, periods, categories


def load_filtered_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Load filtered transaction data with optional caching.

    Args:
        use_cache: Whether to use/create parquet cache file

    Returns:
        Filtered transaction DataFrame
    """
    cache_file = CACHE_DIR / "open_ecommerce_filtered.parquet"

    if use_cache and cache_file.exists():
        print(f"  Loading from cache: {cache_file}")
        return pd.read_parquet(cache_file)

    print("  Loading and filtering raw data...")
    raw_data = load_raw_data()
    filtered = filter_and_preprocess(raw_data)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        filtered.to_parquet(cache_file)
        print(f"  Cached to: {cache_file}")

    return filtered


def get_data_summary(filtered_data: pd.DataFrame) -> dict:
    """
    Get summary statistics for the filtered dataset.

    Args:
        filtered_data: Filtered transaction DataFrame

    Returns:
        Dictionary with summary statistics
    """
    return {
        "n_transactions": len(filtered_data),
        "n_users": filtered_data["user_id"].nunique(),
        "n_periods": filtered_data["period"].nunique(),
        "n_categories": filtered_data["category"].nunique(),
        "total_quantity": filtered_data["quantity"].sum(),
        "total_spend": (filtered_data["quantity"] * filtered_data["unit_price"]).sum(),
        "date_range": f"{filtered_data['period'].min()} to {filtered_data['period'].max()}",
        "avg_price": filtered_data["unit_price"].mean(),
        "categories": sorted(filtered_data["category"].unique().tolist()),
    }


if __name__ == "__main__":
    print("Loading Open E-Commerce 1.0 data...")
    data = load_filtered_data(use_cache=False)
    print("\nSummary:")
    for key, value in get_data_summary(data).items():
        if key == "categories":
            print(f"  {key}: {', '.join(value)}")
        elif isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")
