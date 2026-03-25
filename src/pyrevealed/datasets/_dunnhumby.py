"""Dunnhumby grocery dataset loader.

Loads the Dunnhumby "The Complete Journey" dataset of ~2,500 household
grocery transactions over 104 weeks, returning a BehaviorPanel.

Data must be downloaded separately from Kaggle.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from pyrevealed.core.panel import BehaviorPanel
from pyrevealed.core.session import BehaviorLog

# --- Constants ---

TOP_COMMODITIES = [
    "SOFT DRINKS", "FLUID MILK PRODUCTS", "BAKED BREAD/BUNS/ROLLS",
    "CHEESE", "BAG SNACKS", "SOUP", "YOGURT", "BEEF",
    "FROZEN PIZZA", "LUNCHMEAT",
]

NUM_WEEKS = 104
NUM_PRODUCTS = len(TOP_COMMODITIES)
MIN_UNIT_PRICE = 0.01
MAX_UNIT_PRICE = 50.0


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find dunnhumby data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "dunnhumby")

    candidates.extend([
        Path.home() / ".pyrevealed" / "data" / "dunnhumby",
        Path(__file__).resolve().parents[3] / "dunnhumby" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (d / "transaction_data.csv").exists():
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Dunnhumby data not found. Searched:\n  {searched}\n\n"
        "Download from Kaggle: https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey\n"
        "Then pass data_dir= or set PYREVEALED_DATA_DIR environment variable."
    )


def load_dunnhumby(
    data_dir: str | Path | None = None,
    n_households: int | None = None,
    min_weeks: int = 10,
    period: str | None = None,
) -> BehaviorPanel:
    """Load Dunnhumby grocery dataset as a BehaviorPanel.

    Args:
        data_dir: Path to directory containing transaction_data.csv and
            product.csv. If None, searches standard locations.
        n_households: Max number of households to include (None = all).
        min_weeks: Minimum active shopping weeks per household (default 10).
        period: Time aggregation level. None = one BehaviorLog per household
            across all weeks. "month" = split into monthly sub-sessions.

    Returns:
        BehaviorPanel with one BehaviorLog per household (or household-month).

    Raises:
        FileNotFoundError: If data files cannot be found.
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

    # Load and join
    transactions = pd.read_csv(data_path / "transaction_data.csv")
    products = pd.read_csv(data_path / "product.csv")
    merged = transactions.merge(products[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID")

    # Filter to top commodities
    merged = merged[merged["COMMODITY_DESC"].isin(TOP_COMMODITIES)]

    # Calculate week and unit price
    merged["week"] = ((merged["DAY"] - 1) // 7) + 1
    merged["unit_price"] = (
        (merged["SALES_VALUE"] - merged["RETAIL_DISC"] - merged["COUPON_DISC"])
        / merged["QUANTITY"]
    )
    merged = merged[
        (merged["unit_price"] >= MIN_UNIT_PRICE) &
        (merged["unit_price"] <= MAX_UNIT_PRICE) &
        (merged["QUANTITY"] > 0)
    ]

    # Build price oracle: median price per week per commodity
    price_pivot = merged.pivot_table(
        values="unit_price", index="week", columns="COMMODITY_DESC",
        aggfunc="median",
    ).reindex(index=range(1, NUM_WEEKS + 1), columns=TOP_COMMODITIES)
    price_pivot = price_pivot.ffill().bfill()
    price_grid = price_pivot.values  # (104, 10)

    # Optional month mapping
    if period == "month":
        merged["period"] = ((merged["week"] - 1) // 4) + 1
        group_col = "period"
    else:
        group_col = None

    # Build per-household sessions
    logs: dict[str, BehaviorLog] = {}
    period_map: dict[str, tuple[str, str]] | None = None
    if group_col is not None:
        period_map = {}

    grouped = merged.groupby("household_key")
    hh_keys = list(grouped.groups.keys())

    if n_households is not None:
        hh_keys = hh_keys[:n_households]

    for hh_key in hh_keys:
        hh_data = grouped.get_group(hh_key)

        if group_col is not None:
            # Split by period
            for period_val, period_data in hh_data.groupby(group_col):
                qty_pivot = period_data.pivot_table(
                    values="QUANTITY", index="week", columns="COMMODITY_DESC",
                    aggfunc="sum",
                ).reindex(columns=TOP_COMMODITIES).fillna(0)

                active_weeks = qty_pivot.index.tolist()
                if len(active_weeks) < 2:
                    continue

                qty_matrix = qty_pivot.values
                price_matrix = price_grid[np.array(active_weeks) - 1]  # 0-indexed

                uid = f"household_{hh_key}__period_{period_val}"
                logs[uid] = BehaviorLog(
                    cost_vectors=price_matrix,
                    action_vectors=qty_matrix,
                    user_id=uid,
                )
                period_map[uid] = (f"household_{hh_key}", str(int(period_val)))
        else:
            # All weeks together
            qty_pivot = hh_data.pivot_table(
                values="QUANTITY", index="week", columns="COMMODITY_DESC",
                aggfunc="sum",
            ).reindex(columns=TOP_COMMODITIES).fillna(0)

            active_weeks = qty_pivot[qty_pivot.sum(axis=1) > 0].index.tolist()
            if len(active_weeks) < min_weeks:
                continue

            qty_matrix = qty_pivot.loc[active_weeks].values
            price_matrix = price_grid[np.array(active_weeks) - 1]

            uid = f"household_{hh_key}"
            logs[uid] = BehaviorLog(
                cost_vectors=price_matrix,
                action_vectors=qty_matrix,
                user_id=uid,
            )

    return BehaviorPanel(
        _logs=logs,
        metadata={
            "dataset": "dunnhumby",
            "goods": TOP_COMMODITIES,
            "min_weeks": min_weeks,
            "period": period,
        },
        _period_map=period_map,
    )
