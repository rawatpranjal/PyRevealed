"""Session builder: Create BehaviorLog objects for Open E-Commerce 1.0.

This module transforms user purchase data into the format required
by PyRevealed's revealed preference algorithms.

For each user:
1. Pivot quantities by period and category
2. Align with master price grid
3. Filter to users with sufficient purchase history
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
from config import MIN_OBSERVATIONS
from data_loader import load_filtered_data
from price_oracle import build_price_grid


@dataclass
class UserData:
    """Container for user analysis data."""

    user_id: str
    behavior_log: BehaviorLog
    active_periods: list[str]
    total_spend: float
    num_observations: int


def build_user_quantity_matrix(
    user_data: pd.DataFrame,
    categories: list[str],
) -> Tuple[NDArray[np.float64], list[str]]:
    """
    Build quantity matrix for a single user.

    Pivots transaction-level data into a period x category matrix.

    Args:
        user_data: Filtered transactions for one user
        categories: List of category names (defines column order)

    Returns:
        Tuple of (quantity_matrix, active_periods) where:
        - quantity_matrix: (num_active_periods, num_categories) array
        - active_periods: list of period strings with purchases
    """
    # Aggregate quantities by period and category
    pivot = user_data.pivot_table(
        index="period",
        columns="category",
        values="quantity",
        aggfunc="sum",
        fill_value=0.0,
    )

    # Reindex columns to match category order
    pivot = pivot.reindex(columns=categories, fill_value=0.0)

    # Get active periods (any purchase)
    row_sums = pivot.sum(axis=1)
    active_mask = row_sums > 0
    active_periods = pivot.index[active_mask].tolist()

    # Extract quantity matrix for active periods only
    quantity_matrix = pivot.loc[active_periods].values.astype(np.float64)

    return quantity_matrix, active_periods


def build_single_session(
    user_id: str,
    user_data: pd.DataFrame,
    price_grid: NDArray[np.float64],
    categories: list[str],
    periods: list[str],
    min_observations: int = MIN_OBSERVATIONS,
) -> UserData | None:
    """
    Build BehaviorLog for a single user.

    Args:
        user_id: User identifier
        user_data: Transaction data for this user
        price_grid: Master price grid as numpy array (periods x categories)
        categories: List of category names
        periods: List of period strings (index for price_grid rows)
        min_observations: Minimum observation periods required

    Returns:
        UserData if user qualifies, None otherwise
    """
    # Build quantity matrix
    quantity_matrix, active_periods = build_user_quantity_matrix(user_data, categories)

    # Must have enough observations
    if len(active_periods) < min_observations:
        return None

    # Create period to index mapping
    period_to_idx = {p: i for i, p in enumerate(periods)}

    # Extract corresponding price rows using numpy indexing
    try:
        period_indices = [period_to_idx[p] for p in active_periods]
        price_matrix = price_grid[period_indices, :].astype(np.float64)
    except (KeyError, IndexError):
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
            user_id=f"user_{user_id}",
        )

        total_spend = float(np.sum(behavior_log.total_spend))

        return UserData(
            user_id=str(user_id),
            behavior_log=behavior_log,
            active_periods=active_periods,
            total_spend=total_spend,
            num_observations=behavior_log.num_records,
        )
    except ValueError:
        return None


def build_all_sessions(
    filtered_data: pd.DataFrame,
    price_grid: NDArray[np.float64],
    categories: list[str],
    periods: list[str] | None = None,
    min_observations: int = MIN_OBSERVATIONS,
    progress_interval: int = 500,
) -> Dict[str, UserData]:
    """
    Build BehaviorLog objects for all qualifying users.

    Args:
        filtered_data: Filtered transaction data
        price_grid: Master price grid as numpy array (periods x categories)
        categories: List of category names
        periods: List of period strings (required for numpy price_grid)
        min_observations: Minimum observations required
        progress_interval: Print progress every N users

    Returns:
        Dict mapping user_id -> UserData
    """
    # Get periods from data if not provided
    if periods is None:
        periods = sorted(filtered_data["period"].unique())

    users: Dict[str, UserData] = {}
    skipped_insufficient = 0
    skipped_invalid = 0

    grouped = filtered_data.groupby("user_id")
    total_users = len(grouped)

    for i, (user_id, user_data) in enumerate(grouped):
        result = build_single_session(
            str(user_id), user_data, price_grid, categories, periods, min_observations
        )

        if result is None:
            # Determine skip reason
            qty_matrix, active_periods = build_user_quantity_matrix(user_data, categories)
            if len(active_periods) < min_observations:
                skipped_insufficient += 1
            else:
                skipped_invalid += 1
        else:
            users[str(user_id)] = result

        if (i + 1) % progress_interval == 0:
            print(f"  Processed {i + 1}/{total_users} users...")

    print(f"\n  Session building complete:")
    print(f"    Qualifying users: {len(users)}")
    print(f"    Skipped (< {min_observations} periods): {skipped_insufficient}")
    print(f"    Skipped (invalid data): {skipped_invalid}")

    return users


def get_session_summary(users: Dict[str, UserData]) -> dict:
    """
    Get summary statistics for built sessions.

    Args:
        users: Dict of user data

    Returns:
        Dictionary with summary statistics
    """
    if not users:
        return {"n_users": 0}

    obs_counts = [u.num_observations for u in users.values()]
    spend_totals = [u.total_spend for u in users.values()]

    return {
        "n_users": len(users),
        "total_observations": sum(obs_counts),
        "min_observations": min(obs_counts),
        "max_observations": max(obs_counts),
        "mean_observations": np.mean(obs_counts),
        "median_observations": np.median(obs_counts),
        "total_spend": sum(spend_totals),
        "mean_spend": np.mean(spend_totals),
    }


def load_sessions(use_cache: bool = True) -> Dict[str, UserData]:
    """
    Load or build user sessions.

    Args:
        use_cache: Whether to use cached filtered data

    Returns:
        Dict mapping user_id -> UserData
    """
    print("Loading filtered data...")
    filtered_data = load_filtered_data(use_cache=use_cache)

    print("\nBuilding price oracle...")
    price_grid, periods, categories = build_price_grid(filtered_data, use_cache=use_cache)

    print(f"\nBuilding sessions for {filtered_data['user_id'].nunique()} users...")
    users = build_all_sessions(filtered_data, price_grid, categories, periods)

    return users


if __name__ == "__main__":
    users = load_sessions()
    print("\nSession summary:")
    for key, value in get_session_summary(users).items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value:,}")
