"""Session builder: Create BehaviorLog objects for UCI Online Retail.

This module transforms customer transaction data into the format required
by PyRevealed's revealed preference algorithms.

For each customer:
1. Pivot quantities by period and product
2. Align with master price grid
3. Filter to customers with sufficient shopping history
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pyrevealed import BehaviorLog

sys.path.insert(0, str(Path(__file__).parent))
from config import MIN_TRANSACTIONS
from data_loader import build_price_oracle, load_filtered_data


@dataclass
class CustomerData:
    """Container for customer analysis data."""

    customer_id: int
    behavior_log: BehaviorLog
    active_periods: list[str]
    total_spend: float
    num_observations: int


def build_customer_quantity_matrix(
    customer_data: pd.DataFrame,
    products: list[str],
) -> Tuple[NDArray[np.float64], list[str]]:
    """
    Build quantity matrix for a single customer.

    Pivots transaction-level data into a period x product matrix.

    Args:
        customer_data: Filtered transactions for one customer
        products: List of product codes (defines column order)

    Returns:
        Tuple of (quantity_matrix, active_periods) where:
        - quantity_matrix: (num_active_periods, num_products) array
        - active_periods: list of period strings with purchases
    """
    # Aggregate quantities by period and product
    pivot = customer_data.pivot_table(
        index="period",
        columns="stock_code",
        values="quantity",
        aggfunc="sum",
        fill_value=0.0,
    )

    # Reindex columns to match product order
    pivot = pivot.reindex(columns=products, fill_value=0.0)

    # Get active periods (any purchase)
    row_sums = pivot.sum(axis=1)
    active_mask = row_sums > 0
    active_periods = pivot.index[active_mask].tolist()

    # Extract quantity matrix for active periods only
    quantity_matrix = pivot.loc[active_periods].values.astype(np.float64)

    return quantity_matrix, active_periods


def build_single_session(
    customer_id: int,
    customer_data: pd.DataFrame,
    price_grid: pd.DataFrame,
    products: list[str],
    min_transactions: int = MIN_TRANSACTIONS,
) -> CustomerData | None:
    """
    Build BehaviorLog for a single customer.

    Args:
        customer_id: Customer identifier
        customer_data: Transaction data for this customer
        price_grid: Master price grid (periods x products)
        products: List of product codes
        min_transactions: Minimum transaction count required

    Returns:
        CustomerData if customer qualifies, None otherwise
    """
    # Check minimum transactions
    if len(customer_data) < min_transactions:
        return None

    # Build quantity matrix
    quantity_matrix, active_periods = build_customer_quantity_matrix(
        customer_data, products
    )

    # Must have at least 2 periods for meaningful analysis
    if len(active_periods) < 2:
        return None

    # Extract corresponding price rows
    try:
        price_matrix = price_grid.loc[active_periods, products].values.astype(np.float64)
    except KeyError:
        return None

    # Validate: prices must be strictly positive
    if np.any(price_matrix <= 0):
        return None

    # Validate: quantities must be non-negative
    if np.any(quantity_matrix < 0):
        return None

    # Create BehaviorLog
    try:
        behavior_log = BehaviorLog(
            cost_vectors=price_matrix,
            action_vectors=quantity_matrix,
            user_id=f"customer_{customer_id}",
        )

        total_spend = float(np.sum(behavior_log.total_spend))

        return CustomerData(
            customer_id=customer_id,
            behavior_log=behavior_log,
            active_periods=active_periods,
            total_spend=total_spend,
            num_observations=behavior_log.num_records,
        )
    except ValueError:
        return None


def build_all_sessions(
    filtered_data: pd.DataFrame,
    price_grid: pd.DataFrame,
    products: list[str],
    min_transactions: int = MIN_TRANSACTIONS,
    progress_interval: int = 500,
) -> Dict[int, CustomerData]:
    """
    Build BehaviorLog objects for all qualifying customers.

    Args:
        filtered_data: Filtered transaction data
        price_grid: Master price grid (periods x products)
        products: List of product codes
        min_transactions: Minimum transactions required
        progress_interval: Print progress every N customers

    Returns:
        Dict mapping customer_id -> CustomerData
    """
    customers: Dict[int, CustomerData] = {}
    skipped_insufficient = 0
    skipped_invalid = 0

    grouped = filtered_data.groupby("customer_id")
    total_customers = len(grouped)

    for i, (customer_id, customer_data) in enumerate(grouped):
        result = build_single_session(
            customer_id, customer_data, price_grid, products, min_transactions
        )

        if result is None:
            if len(customer_data) < min_transactions:
                skipped_insufficient += 1
            else:
                skipped_invalid += 1
        else:
            customers[customer_id] = result

        if (i + 1) % progress_interval == 0:
            print(f"  Processed {i + 1}/{total_customers} customers...")

    print(f"\n  Session building complete:")
    print(f"    Qualifying customers: {len(customers)}")
    print(f"    Skipped (< {min_transactions} transactions): {skipped_insufficient}")
    print(f"    Skipped (invalid data): {skipped_invalid}")

    return customers


def get_session_summary(customers: Dict[int, CustomerData]) -> dict:
    """
    Get summary statistics for built sessions.

    Args:
        customers: Dict of customer data

    Returns:
        Dictionary with summary statistics
    """
    if not customers:
        return {"n_customers": 0}

    obs_counts = [c.num_observations for c in customers.values()]
    spend_totals = [c.total_spend for c in customers.values()]

    return {
        "n_customers": len(customers),
        "total_observations": sum(obs_counts),
        "min_observations": min(obs_counts),
        "max_observations": max(obs_counts),
        "mean_observations": np.mean(obs_counts),
        "median_observations": np.median(obs_counts),
        "total_spend": sum(spend_totals),
        "mean_spend": np.mean(spend_totals),
    }


def load_sessions(use_cache: bool = True) -> Dict[int, CustomerData]:
    """
    Load or build customer sessions.

    Args:
        use_cache: Whether to use cached filtered data

    Returns:
        Dict mapping customer_id -> CustomerData
    """
    print("Loading filtered data...")
    filtered_data = load_filtered_data(use_cache=use_cache)

    print("\nBuilding price oracle...")
    price_grid, periods, products = build_price_oracle(filtered_data)

    print(f"\nBuilding sessions for {filtered_data['customer_id'].nunique()} customers...")
    customers = build_all_sessions(filtered_data, price_grid, products)

    return customers


if __name__ == "__main__":
    customers = load_sessions()
    print("\nSession summary:")
    for key, value in get_session_summary(customers).items():
        print(f"  {key}: {value}")
