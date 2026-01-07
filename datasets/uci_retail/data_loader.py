"""Data loading and preprocessing for UCI Online Retail dataset.

This module handles:
1. Loading the Excel file
2. Parsing dates and creating time periods (monthly)
3. Filtering cancelled orders and invalid data
4. Selecting top products by transaction volume
5. Computing price oracle (median prices per period)
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
    DATA_FILE,
    MAX_UNIT_PRICE,
    MIN_TRANSACTIONS,
    MIN_UNIT_PRICE,
    TOP_N_PRODUCTS,
)


def load_raw_data() -> pd.DataFrame:
    """
    Load raw transaction data from Excel file.

    Returns:
        Raw transactions DataFrame

    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_FILE}\n"
            "Please run download.py first to download the dataset."
        )

    print(f"  Loading from: {DATA_FILE}")
    df = pd.read_excel(DATA_FILE)
    print(f"  Raw transactions: {len(df):,}")

    return df


def filter_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and preprocess transaction data.

    Steps:
    1. Remove cancelled orders (InvoiceNo starting with 'C')
    2. Remove rows with missing CustomerID
    3. Remove invalid prices and quantities
    4. Parse dates and create monthly period
    5. Select top N products by volume

    Args:
        df: Raw transaction DataFrame

    Returns:
        Filtered DataFrame with columns:
        - customer_id: Customer identifier
        - period: Year-month string (e.g., "2011-01")
        - stock_code: Product code
        - quantity: Units purchased
        - unit_price: Price per unit (GBP)
    """
    # Copy to avoid modifying original
    df = df.copy()

    # 1. Remove cancelled orders
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    is_cancelled = df["InvoiceNo"].str.startswith("C")
    n_cancelled = is_cancelled.sum()
    df = df[~is_cancelled]
    print(f"  Removed {n_cancelled:,} cancelled orders")

    # 2. Remove missing CustomerID
    n_missing_customer = df["CustomerID"].isna().sum()
    df = df[df["CustomerID"].notna()]
    print(f"  Removed {n_missing_customer:,} rows with missing CustomerID")

    # 3. Filter invalid quantities
    valid_qty = df["Quantity"] > 0
    n_invalid_qty = (~valid_qty).sum()
    df = df[valid_qty]
    print(f"  Removed {n_invalid_qty:,} rows with non-positive quantity")

    # 4. Filter invalid prices
    valid_price = (df["UnitPrice"] >= MIN_UNIT_PRICE) & (df["UnitPrice"] <= MAX_UNIT_PRICE)
    n_invalid_price = (~valid_price).sum()
    df = df[valid_price]
    print(f"  Removed {n_invalid_price:,} rows with invalid prices")

    # 5. Parse dates and create monthly period
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["period"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # 6. Select top N products by transaction count
    product_counts = df["StockCode"].value_counts()
    top_products = product_counts.head(TOP_N_PRODUCTS).index.tolist()
    df = df[df["StockCode"].isin(top_products)]
    print(f"  Filtered to top {TOP_N_PRODUCTS} products: {len(df):,} transactions")

    # 7. Rename and select columns
    result = df[["CustomerID", "period", "StockCode", "Quantity", "UnitPrice"]].rename(
        columns={
            "CustomerID": "customer_id",
            "StockCode": "stock_code",
            "Quantity": "quantity",
            "UnitPrice": "unit_price",
        }
    )

    # Convert types for consistent serialization
    result["customer_id"] = result["customer_id"].astype(int)
    result["stock_code"] = result["stock_code"].astype(str)  # Ensure string type

    print(f"  Final transactions: {len(result):,}")
    print(f"  Unique customers: {result['customer_id'].nunique():,}")
    print(f"  Unique periods: {result['period'].nunique()}")
    print(f"  Unique products: {result['stock_code'].nunique()}")

    return result


def build_price_oracle(
    filtered_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, list[str], list[str]]:
    """
    Build the master price grid (periods x products).

    For each period and product, compute the median price across all customers.
    Forward/backward fill any gaps.

    Args:
        filtered_data: Filtered transaction DataFrame

    Returns:
        Tuple of:
        - price_grid: DataFrame with periods as index, products as columns
        - periods: List of period strings (sorted)
        - products: List of product codes (sorted)
    """
    # Get unique periods and products (sorted)
    periods = sorted(filtered_data["period"].unique())
    products = sorted(filtered_data["stock_code"].unique())

    print(f"  Building price grid: {len(periods)} periods x {len(products)} products")

    # Compute median price per period per product
    price_pivot = filtered_data.pivot_table(
        index="period",
        columns="stock_code",
        values="unit_price",
        aggfunc="median",
    )

    # Reindex to ensure all periods and products are present
    price_pivot = price_pivot.reindex(index=periods, columns=products)

    # Forward fill then backward fill
    price_pivot = price_pivot.ffill().bfill()

    # Check for any remaining NaN (shouldn't happen after bfill)
    n_nan = price_pivot.isna().sum().sum()
    if n_nan > 0:
        print(f"  Warning: {n_nan} missing prices after fill, using global median")
        for col in price_pivot.columns:
            if price_pivot[col].isna().any():
                global_median = filtered_data.loc[
                    filtered_data["stock_code"] == col, "unit_price"
                ].median()
                price_pivot[col] = price_pivot[col].fillna(global_median)

    return price_pivot, periods, products


def load_filtered_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Load filtered transaction data with optional caching.

    Args:
        use_cache: Whether to use/create parquet cache file

    Returns:
        Filtered transaction DataFrame
    """
    cache_file = CACHE_DIR / "uci_retail_filtered.parquet"

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
        "n_customers": filtered_data["customer_id"].nunique(),
        "n_periods": filtered_data["period"].nunique(),
        "n_products": filtered_data["stock_code"].nunique(),
        "total_quantity": filtered_data["quantity"].sum(),
        "total_spend": (filtered_data["quantity"] * filtered_data["unit_price"]).sum(),
        "date_range": f"{filtered_data['period'].min()} to {filtered_data['period'].max()}",
        "avg_price": filtered_data["unit_price"].mean(),
    }


if __name__ == "__main__":
    print("Loading UCI Online Retail data...")
    data = load_filtered_data(use_cache=False)
    print("\nSummary:")
    for key, value in get_data_summary(data).items():
        print(f"  {key}: {value}")
