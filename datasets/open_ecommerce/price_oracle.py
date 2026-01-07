"""Price oracle construction for Open E-Commerce 1.0 dataset (Phase 2).

This module builds the master price grid that provides opportunity costs
for revealed preference analysis. The challenge with e-commerce data is
that we only observe prices of items purchased, not rejected alternatives.

Solution: Build a (periods × categories) matrix using median prices
across all users for each time period.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent))

from config import CACHE_DIR


def build_price_grid(
    filtered_data: pd.DataFrame,
    use_cache: bool = True,
) -> Tuple[NDArray[np.float64], list[str], list[str]]:
    """
    Build the master price grid (periods × categories).

    For each period and category, compute the median price across all users.
    Forward/backward fill any gaps to ensure complete coverage.

    Args:
        filtered_data: Filtered transaction DataFrame with columns:
            - user_id, period, category, quantity, unit_price
        use_cache: Whether to use/create numpy cache file

    Returns:
        Tuple of:
        - price_grid: (num_periods, num_categories) numpy array
        - periods: List of period strings (sorted)
        - categories: List of category names (sorted)
    """
    cache_file = CACHE_DIR / "open_ecommerce_price_grid.npy"
    periods_file = CACHE_DIR / "open_ecommerce_periods.npy"
    categories_file = CACHE_DIR / "open_ecommerce_categories.npy"

    # Check cache
    if use_cache and cache_file.exists() and periods_file.exists() and categories_file.exists():
        print("  Loading price grid from cache...")
        price_grid = np.load(cache_file)
        periods = np.load(periods_file, allow_pickle=True).tolist()
        categories = np.load(categories_file, allow_pickle=True).tolist()
        return price_grid, periods, categories

    # Get unique periods and categories (sorted)
    periods = sorted(filtered_data["period"].unique())
    categories = sorted(filtered_data["category"].unique())

    print(f"  Building price grid: {len(periods)} periods × {len(categories)} categories")

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
                if pd.isna(global_median):
                    global_median = filtered_data["unit_price"].median()
                price_pivot[col] = price_pivot[col].fillna(global_median)

    price_grid = price_pivot.values.astype(np.float64)

    # Cache results
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, price_grid)
        np.save(periods_file, np.array(periods, dtype=object))
        np.save(categories_file, np.array(categories, dtype=object))
        print(f"  Cached to: {cache_file}")

    return price_grid, periods, categories


def get_price_grid_summary(
    price_grid: NDArray[np.float64],
    periods: list[str],
    categories: list[str],
) -> dict:
    """
    Get summary statistics for the price grid.

    Args:
        price_grid: (periods × categories) numpy array
        periods: List of period strings
        categories: List of category names

    Returns:
        Dictionary with summary statistics
    """
    return {
        "shape": price_grid.shape,
        "n_periods": len(periods),
        "n_categories": len(categories),
        "date_range": f"{periods[0]} to {periods[-1]}",
        "min_price": float(np.min(price_grid)),
        "max_price": float(np.max(price_grid)),
        "mean_price": float(np.mean(price_grid)),
        "median_price": float(np.median(price_grid)),
        "std_price": float(np.std(price_grid)),
    }


def print_price_grid_summary(
    price_grid: NDArray[np.float64],
    periods: list[str],
    categories: list[str],
) -> None:
    """Print formatted price grid summary."""
    summary = get_price_grid_summary(price_grid, periods, categories)
    print(f"  Shape: {summary['shape'][0]} periods × {summary['shape'][1]} categories")
    print(f"  Date range: {summary['date_range']}")
    print(f"  Price range: ${summary['min_price']:.2f} - ${summary['max_price']:.2f}")
    print(f"  Mean price: ${summary['mean_price']:.2f}")


if __name__ == "__main__":
    from data_loader import load_filtered_data

    print("Building price oracle for Open E-Commerce 1.0...")
    filtered_data = load_filtered_data(use_cache=True)
    price_grid, periods, categories = build_price_grid(filtered_data, use_cache=False)

    print("\nPrice Grid Summary:")
    print_price_grid_summary(price_grid, periods, categories)

    print(f"\nTop 10 categories: {categories[:10]}")
