"""Price Oracle: Master price grid construction (Phase 2).

The Problem:
    Raw data only shows the price of items *bought*. To run revealed preference
    analysis, we need the price of items *not bought* (the opportunity cost).

The Solution:
    Construct a Master_Price_Grid containing the "market price" for every
    commodity in every week. This uses the median price paid across all
    stores/users for that commodity-week combination.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from config import CACHE_DIR, NUM_PRODUCTS, NUM_WEEKS, TOP_COMMODITIES


def build_master_price_grid(filtered_data: pd.DataFrame) -> NDArray[np.float64]:
    """
    Build master price grid: median price per commodity per week.

    The price grid represents the "market price" for each commodity in each week.
    Using median (rather than mean) provides robustness against outlier transactions.

    Args:
        filtered_data: Filtered transaction DataFrame from data_loader

    Returns:
        (104, 10) array where:
        - rows = weeks (0-103 for weeks 1-104)
        - cols = commodities (in TOP_COMMODITIES order)

    Raises:
        ValueError: If resulting grid has unexpected shape
    """
    # Aggregate to median price per commodity per week
    weekly_prices = (
        filtered_data.groupby(["week", "commodity"])["unit_price"]
        .median()
        .unstack(fill_value=np.nan)
    )

    # Reindex to ensure all weeks present (1 to NUM_WEEKS)
    all_weeks = pd.Index(range(1, NUM_WEEKS + 1), name="week")
    weekly_prices = weekly_prices.reindex(all_weeks)

    # Reindex columns to match TOP_COMMODITIES order
    weekly_prices = weekly_prices.reindex(columns=TOP_COMMODITIES)

    # Forward-fill missing values (use previous week's price)
    weekly_prices = weekly_prices.ffill()

    # Backward-fill any remaining NaNs at the start
    weekly_prices = weekly_prices.bfill()

    # Check for any remaining NaNs
    n_nan = weekly_prices.isna().sum().sum()
    if n_nan > 0:
        raise ValueError(f"Price grid still has {n_nan} NaN values after fill")

    # Convert to numpy array
    price_grid = weekly_prices.values.astype(np.float64)

    # Validate shape
    expected_shape = (NUM_WEEKS, NUM_PRODUCTS)
    if price_grid.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {price_grid.shape}")

    return price_grid


def get_master_price_grid(
    filtered_data: pd.DataFrame,
    use_cache: bool = True,
) -> NDArray[np.float64]:
    """
    Get master price grid with optional numpy file caching.

    Args:
        filtered_data: Filtered transaction DataFrame
        use_cache: Whether to use/create .npy cache file

    Returns:
        (104, 10) master price grid
    """
    cache_file = CACHE_DIR / "master_price_grid.npy"

    if use_cache and cache_file.exists():
        print(f"  Loading from cache: {cache_file}")
        return np.load(cache_file)

    print("  Building master price grid...")
    price_grid = build_master_price_grid(filtered_data)

    if use_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(cache_file, price_grid)
        print(f"  Cached to: {cache_file}")

    return price_grid


def get_price_grid_summary(price_grid: NDArray[np.float64]) -> dict:
    """
    Get summary statistics for the price grid.

    Args:
        price_grid: (104, 10) master price grid

    Returns:
        Dictionary with summary statistics
    """
    return {
        "shape": price_grid.shape,
        "min_price": float(price_grid.min()),
        "max_price": float(price_grid.max()),
        "mean_price": float(price_grid.mean()),
        "std_price": float(price_grid.std()),
        "price_by_commodity": {
            TOP_COMMODITIES[i]: {
                "min": float(price_grid[:, i].min()),
                "max": float(price_grid[:, i].max()),
                "mean": float(price_grid[:, i].mean()),
            }
            for i in range(NUM_PRODUCTS)
        },
    }


def validate_price_grid(price_grid: NDArray[np.float64]) -> bool:
    """
    Validate that price grid is suitable for revealed preference analysis.

    Checks:
    - All prices are strictly positive
    - No NaN or infinite values
    - Reasonable price range
    - Sufficient price variation over time

    Args:
        price_grid: (104, 10) master price grid

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check for NaN/inf
    if np.any(np.isnan(price_grid)):
        raise ValueError("Price grid contains NaN values")
    if np.any(np.isinf(price_grid)):
        raise ValueError("Price grid contains infinite values")

    # Check strictly positive
    if np.any(price_grid <= 0):
        raise ValueError("All prices must be strictly positive")

    # Check reasonable range (per-unit prices)
    if price_grid.min() < 0.001:
        raise ValueError(f"Price too low: {price_grid.min()}")
    if price_grid.max() > 100:
        raise ValueError(f"Price too high: {price_grid.max()}")

    # Check for price variation (prices shouldn't be constant)
    for i in range(NUM_PRODUCTS):
        col_std = np.std(price_grid[:, i])
        if col_std < 0.001:
            print(f"  Warning: {TOP_COMMODITIES[i]} has very low price variation")

    return True
