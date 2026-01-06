"""Data loading and filtering for Dunnhumby analysis (Phase 1).

This module handles:
1. Loading transaction_data.csv and product.csv
2. Joining on PRODUCT_ID to get commodity information
3. Filtering to TOP_COMMODITIES (staple goods)
4. Calculating unit price after discounts
5. Converting days to week numbers
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    CACHE_DIR,
    MAX_UNIT_PRICE,
    MIN_UNIT_PRICE,
    PRODUCT_FILE,
    TOP_COMMODITIES,
    TRANSACTION_FILE,
)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw transaction and product data from CSV files.

    Returns:
        Tuple of (transactions_df, products_df)

    Raises:
        FileNotFoundError: If data files don't exist (run download_data.sh first)
    """
    if not TRANSACTION_FILE.exists():
        raise FileNotFoundError(
            f"Transaction file not found: {TRANSACTION_FILE}\n"
            "Please run download_data.sh first to download the Kaggle dataset."
        )
    if not PRODUCT_FILE.exists():
        raise FileNotFoundError(
            f"Product file not found: {PRODUCT_FILE}\n"
            "Please run download_data.sh first to download the Kaggle dataset."
        )

    transactions = pd.read_csv(TRANSACTION_FILE)
    products = pd.read_csv(PRODUCT_FILE)

    return transactions, products


def calculate_week_number(day: int) -> int:
    """
    Convert day number (1-728) to week number (1-104).

    The Dunnhumby dataset uses day numbers from 1 to ~728 (2 years).
    We aggregate to weeks to reduce noise while capturing weekly flyer cycles.

    Args:
        day: Day number (1-based)

    Returns:
        Week number (1-based, 1-104)
    """
    return ((day - 1) // 7) + 1


def calculate_unit_price(
    sales_value: float,
    retail_disc: float,
    coupon_disc: float,
    quantity: float,
) -> float:
    """
    Calculate effective unit price after all discounts.

    Formula: FINAL_PRICE = (SALES_VALUE - RETAIL_DISC - COUPON_DISC) / QUANTITY

    Args:
        sales_value: Gross sales amount
        retail_disc: Retail discount (positive value)
        coupon_disc: Coupon discount (positive value)
        quantity: Number of units purchased

    Returns:
        Unit price, or NaN if quantity <= 0
    """
    if quantity <= 0:
        return np.nan
    net_sales = sales_value - retail_disc - coupon_disc
    return net_sales / quantity


def filter_and_preprocess(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter transactions to top commodities and calculate derived fields.

    Steps:
    1. Join transactions with products on PRODUCT_ID
    2. Filter to TOP_COMMODITIES only
    3. Calculate week number from day
    4. Calculate unit price after discounts
    5. Remove invalid prices (zero, negative, outliers)

    Args:
        transactions: Raw transaction DataFrame
        products: Raw product DataFrame

    Returns:
        Filtered DataFrame with columns:
        - household_key: Household identifier
        - week: Week number (1-104)
        - commodity: Product category name
        - quantity: Units purchased
        - unit_price: Price per unit after discounts
        - store_id: Store identifier
    """
    # Merge transactions with products to get commodity info
    merged = transactions.merge(
        products[["PRODUCT_ID", "COMMODITY_DESC"]],
        on="PRODUCT_ID",
        how="left",
    )

    # Filter to top commodities only
    merged = merged[merged["COMMODITY_DESC"].isin(TOP_COMMODITIES)].copy()

    print(f"  Transactions in top commodities: {len(merged):,}")

    # Calculate week number
    merged["week"] = merged["DAY"].apply(calculate_week_number)

    # Calculate unit price (vectorized)
    merged["unit_price"] = (
        merged["SALES_VALUE"] - merged["RETAIL_DISC"] - merged["COUPON_DISC"]
    ) / merged["QUANTITY"]

    # Filter out invalid prices
    valid_prices = (
        (merged["unit_price"] > MIN_UNIT_PRICE)
        & (merged["unit_price"] < MAX_UNIT_PRICE)
        & merged["unit_price"].notna()
    )
    n_invalid = (~valid_prices).sum()
    if n_invalid > 0:
        print(f"  Removed {n_invalid:,} transactions with invalid prices")
    merged = merged[valid_prices].copy()

    # Filter out zero/negative quantities
    valid_qty = merged["QUANTITY"] > 0
    n_invalid_qty = (~valid_qty).sum()
    if n_invalid_qty > 0:
        print(f"  Removed {n_invalid_qty:,} transactions with invalid quantities")
    merged = merged[valid_qty].copy()

    # Select and rename columns
    result = merged[
        [
            "household_key",
            "week",
            "COMMODITY_DESC",
            "QUANTITY",
            "unit_price",
            "STORE_ID",
        ]
    ].rename(
        columns={
            "COMMODITY_DESC": "commodity",
            "QUANTITY": "quantity",
            "STORE_ID": "store_id",
        }
    )

    return result


def load_filtered_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Load filtered transaction data, with optional parquet caching.

    Caching significantly speeds up repeated runs (parquet is much faster
    than re-processing the full CSV).

    Args:
        use_cache: Whether to use/create parquet cache file

    Returns:
        Filtered transaction DataFrame
    """
    cache_file = CACHE_DIR / "filtered_transactions.parquet"

    if use_cache and cache_file.exists():
        print(f"  Loading from cache: {cache_file}")
        return pd.read_parquet(cache_file)

    print("  Loading and filtering raw data...")
    transactions, products = load_raw_data()

    print(f"  Raw transactions: {len(transactions):,}")
    print(f"  Products in catalog: {len(products):,}")

    filtered = filter_and_preprocess(transactions, products)

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
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
        "n_households": filtered_data["household_key"].nunique(),
        "n_weeks": filtered_data["week"].nunique(),
        "n_commodities": filtered_data["commodity"].nunique(),
        "n_stores": filtered_data["store_id"].nunique(),
        "total_quantity": filtered_data["quantity"].sum(),
        "total_spend": (filtered_data["quantity"] * filtered_data["unit_price"]).sum(),
        "avg_price_by_commodity": filtered_data.groupby("commodity")["unit_price"]
        .mean()
        .to_dict(),
    }
