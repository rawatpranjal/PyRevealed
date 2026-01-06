"""Session builder: Create BehaviorLog objects for each household (Phase 3).

This module transforms household transaction data into the format required
by PyRevealed's revealed preference algorithms.

For each household:
1. Pivot quantities by week and commodity
2. Align with master price grid
3. Filter to "active weeks" (weeks with purchases)
4. Only include households with sufficient shopping history
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pyrevealed import BehaviorLog

from config import MIN_SHOPPING_WEEKS, NUM_PRODUCTS, TOP_COMMODITIES


@dataclass
class HouseholdData:
    """Container for household analysis data."""

    household_key: int
    behavior_log: BehaviorLog
    active_weeks: list[int]
    total_spend: float
    num_observations: int


def build_household_quantity_matrix(
    household_data: pd.DataFrame,
) -> Tuple[NDArray[np.float64], list[int]]:
    """
    Build quantity matrix for a single household.

    Pivots transaction-level data into a week x commodity matrix.

    Args:
        household_data: Filtered transactions for one household

    Returns:
        Tuple of (quantity_matrix, active_weeks) where:
        - quantity_matrix: (num_active_weeks, 10) array of quantities
        - active_weeks: list of week numbers with purchases
    """
    # Pivot: rows=weeks, columns=commodities, values=sum of quantities
    pivot = household_data.pivot_table(
        index="week",
        columns="commodity",
        values="quantity",
        aggfunc="sum",
        fill_value=0.0,
    )

    # Reindex columns to match TOP_COMMODITIES order
    pivot = pivot.reindex(columns=TOP_COMMODITIES, fill_value=0.0)

    # Get active weeks (non-zero total quantity)
    row_sums = pivot.sum(axis=1)
    active_mask = row_sums > 0
    active_weeks = pivot.index[active_mask].tolist()

    # Extract quantity matrix for active weeks only
    quantity_matrix = pivot.loc[active_weeks].values.astype(np.float64)

    return quantity_matrix, active_weeks


def build_single_session(
    hh_key: int,
    hh_data: pd.DataFrame,
    price_grid: NDArray[np.float64],
    min_weeks: int = MIN_SHOPPING_WEEKS,
) -> HouseholdData | None:
    """
    Build BehaviorLog for a single household.

    Args:
        hh_key: Household identifier
        hh_data: Transaction data for this household
        price_grid: Master price grid (104, 10)
        min_weeks: Minimum shopping weeks required

    Returns:
        HouseholdData if household qualifies, None otherwise
    """
    # Build quantity matrix
    quantity_matrix, active_weeks = build_household_quantity_matrix(hh_data)

    # Filter: must have enough shopping weeks
    if len(active_weeks) < min_weeks:
        return None

    # Extract corresponding price rows (week indices are 1-based)
    week_indices = [w - 1 for w in active_weeks]  # Convert to 0-based
    price_matrix = price_grid[week_indices, :].copy()

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
            user_id=f"household_{hh_key}",
        )

        total_spend = float(np.sum(behavior_log.total_spend))

        return HouseholdData(
            household_key=hh_key,
            behavior_log=behavior_log,
            active_weeks=active_weeks,
            total_spend=total_spend,
            num_observations=behavior_log.num_records,
        )
    except ValueError:
        # Skip households with invalid data (e.g., all zeros)
        return None


def build_all_sessions(
    filtered_data: pd.DataFrame,
    price_grid: NDArray[np.float64],
    min_weeks: int = MIN_SHOPPING_WEEKS,
    progress_interval: int = 500,
) -> Dict[int, HouseholdData]:
    """
    Build BehaviorLog objects for all qualifying households.

    Args:
        filtered_data: Filtered transaction data
        price_grid: Master price grid (104, 10)
        min_weeks: Minimum shopping weeks required
        progress_interval: Print progress every N households

    Returns:
        Dict mapping household_key -> HouseholdData
    """
    households: Dict[int, HouseholdData] = {}
    skipped_insufficient_weeks = 0
    skipped_invalid_data = 0

    grouped = filtered_data.groupby("household_key")
    total_households = len(grouped)

    for i, (hh_key, hh_data) in enumerate(grouped):
        result = build_single_session(hh_key, hh_data, price_grid, min_weeks)

        if result is None:
            # Determine skip reason
            qty_matrix, active_weeks = build_household_quantity_matrix(hh_data)
            if len(active_weeks) < min_weeks:
                skipped_insufficient_weeks += 1
            else:
                skipped_invalid_data += 1
        else:
            households[hh_key] = result

        # Progress reporting
        if (i + 1) % progress_interval == 0:
            print(f"  Processed {i + 1}/{total_households} households...")

    print(f"\n  Session building complete:")
    print(f"    Qualifying households: {len(households)}")
    print(f"    Skipped (< {min_weeks} weeks): {skipped_insufficient_weeks}")
    print(f"    Skipped (invalid data): {skipped_invalid_data}")

    return households


def get_session_summary(households: Dict[int, HouseholdData]) -> dict:
    """
    Get summary statistics for built sessions.

    Args:
        households: Dict of household data

    Returns:
        Dictionary with summary statistics
    """
    if not households:
        return {"n_households": 0}

    obs_counts = [h.num_observations for h in households.values()]
    spend_totals = [h.total_spend for h in households.values()]

    return {
        "n_households": len(households),
        "total_observations": sum(obs_counts),
        "min_observations": min(obs_counts),
        "max_observations": max(obs_counts),
        "mean_observations": np.mean(obs_counts),
        "median_observations": np.median(obs_counts),
        "total_spend": sum(spend_totals),
        "mean_spend": np.mean(spend_totals),
    }
