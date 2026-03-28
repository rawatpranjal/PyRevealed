"""M5 Walmart Forecasting dataset loader.

Loads the M5 Forecasting competition dataset of 30,490 items across 10
Walmart stores over ~1,941 days (~278 weeks), returning a BehaviorPanel.

Aggregates item-level daily sales into weekly store-level (or
store-department-level) price-quantity panels for budget-based revealed
preference analysis.

Data must be downloaded separately from Kaggle.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from prefgraph.core.panel import BehaviorPanel
from prefgraph.core.session import BehaviorLog

# --- Constants ---

DEPARTMENTS = [
    "FOODS_1", "FOODS_2", "FOODS_3",
    "HOBBIES_1", "HOBBIES_2",
    "HOUSEHOLD_1", "HOUSEHOLD_2",
]

CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]

NUM_DAYS = 1941
MIN_UNIT_PRICE = 0.01


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find M5 data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "m5")

    candidates.extend([
        Path.home() / ".prefgraph" / "data" / "m5",
        Path(__file__).resolve().parents[3] / "m5" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (d / "sales_train_evaluation.csv").exists():
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"M5 data not found. Searched:\n  {searched}\n\n"
        "Download from Kaggle: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data\n"
        "Place sales_train_evaluation.csv, sell_prices.csv, and calendar.csv "
        "in the data directory.\n"
        "Then pass data_dir= or set PYREVEALED_DATA_DIR environment variable."
    )


def load_m5(
    data_dir: str | Path | None = None,
    aggregation: str = "store",
    min_weeks: int = 100,
    max_users: int | None = None,
) -> BehaviorPanel:
    """Load M5 Walmart dataset as a BehaviorPanel.

    Constructs weekly price-quantity panels by aggregating daily item-level
    sales within each store (or store-department). Quantities are total units
    sold per department (or category) per week. Prices are mean sell_price
    for that department-store-week.

    Args:
        data_dir: Path to directory containing sales_train_evaluation.csv,
            sell_prices.csv, and calendar.csv. If None, searches standard
            locations.
        aggregation: User granularity level.

            - "store": 10 users (one per store), 7 goods (departments).
            - "store_dept": 70 users (one per store-department combination),
              goods are items within that department (aggregated to category).
              Each user has 3 goods (FOODS, HOBBIES, HOUSEHOLD).
        min_weeks: Minimum weeks with nonzero sales required per user
            (default 100).
        max_users: Optional cap on number of users returned.

    Returns:
        BehaviorPanel with one BehaviorLog per store (or store-dept).

    Raises:
        FileNotFoundError: If data files cannot be found.
        ImportError: If pandas is not installed.
        ValueError: If aggregation is not "store" or "store_dept".
    """
    if aggregation not in ("store", "store_dept"):
        raise ValueError(
            f"aggregation must be 'store' or 'store_dept', got '{aggregation}'"
        )

    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for dataset loaders. "
            "Install with: pip install 'prefgraph[datasets]'"
        ) from None

    data_path = _find_data_dir(data_dir)

    # --- Load calendar: map d_* columns to wm_yr_wk ---
    calendar = pd.read_csv(data_path / "calendar.csv", usecols=["d", "wm_yr_wk"])
    # Only keep days within the sales range (d_1 .. d_1941)
    calendar["day_num"] = calendar["d"].str.extract(r"d_(\d+)").astype(int)
    calendar = calendar[calendar["day_num"] <= NUM_DAYS]
    d_to_week = dict(zip(calendar["d"], calendar["wm_yr_wk"]))
    weeks_sorted = sorted(calendar["wm_yr_wk"].unique())

    # --- Load sell_prices ---
    prices_df = pd.read_csv(data_path / "sell_prices.csv")

    # --- Load sales (wide format) ---
    sales = pd.read_csv(data_path / "sales_train_evaluation.csv")
    d_cols = [f"d_{i}" for i in range(1, NUM_DAYS + 1)]

    # Melt to long format: one row per item-day
    id_cols = ["item_id", "dept_id", "cat_id", "store_id"]
    sales_long = sales[id_cols + d_cols].melt(
        id_vars=id_cols, var_name="d", value_name="units",
    )
    sales_long["wm_yr_wk"] = sales_long["d"].map(d_to_week)
    sales_long.dropna(subset=["wm_yr_wk"], inplace=True)
    sales_long["wm_yr_wk"] = sales_long["wm_yr_wk"].astype(int)

    # Merge prices
    sales_long = sales_long.merge(
        prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left",
    )

    # Drop rows with missing prices or zero units
    sales_long = sales_long[
        sales_long["sell_price"].notna() &
        (sales_long["sell_price"] >= MIN_UNIT_PRICE)
    ]

    if aggregation == "store":
        return _build_store_panel(
            sales_long, weeks_sorted, min_weeks, max_users,
        )
    else:
        return _build_store_dept_panel(
            sales_long, weeks_sorted, min_weeks, max_users,
        )


def _build_store_panel(
    sales_long,
    weeks_sorted: list[int],
    min_weeks: int,
    max_users: int | None,
) -> BehaviorPanel:
    """Build panel with one user per store, departments as goods."""
    import pandas as pd

    week_index = {w: i for i, w in enumerate(weeks_sorted)}

    # Aggregate: total units and mean price per store-dept-week
    agg = sales_long.groupby(["store_id", "dept_id", "wm_yr_wk"]).agg(
        units=("units", "sum"),
        price=("sell_price", "mean"),
    ).reset_index()

    stores = sorted(agg["store_id"].unique())
    if max_users is not None:
        stores = stores[:max_users]

    logs: dict[str, BehaviorLog] = {}

    for store in stores:
        store_data = agg[agg["store_id"] == store]

        # Pivot quantities: rows=weeks, columns=departments
        qty_pivot = store_data.pivot_table(
            values="units", index="wm_yr_wk", columns="dept_id",
            aggfunc="sum",
        ).reindex(index=weeks_sorted, columns=DEPARTMENTS).fillna(0)

        # Pivot prices: rows=weeks, columns=departments
        price_pivot = store_data.pivot_table(
            values="price", index="wm_yr_wk", columns="dept_id",
            aggfunc="mean",
        ).reindex(index=weeks_sorted, columns=DEPARTMENTS)
        price_pivot = price_pivot.ffill().bfill()

        # Keep weeks where at least one department has sales
        active_mask = qty_pivot.sum(axis=1) > 0
        active_weeks = qty_pivot.index[active_mask].tolist()

        if len(active_weeks) < min_weeks:
            continue

        qty_matrix = qty_pivot.loc[active_weeks].values.astype(np.float64)
        price_matrix = price_pivot.loc[active_weeks].values.astype(np.float64)

        # Fill any remaining NaN prices with column median
        for col in range(price_matrix.shape[1]):
            mask = np.isnan(price_matrix[:, col])
            if mask.any():
                median_val = np.nanmedian(price_matrix[:, col])
                price_matrix[mask, col] = median_val if not np.isnan(median_val) else 1.0

        uid = store
        logs[uid] = BehaviorLog(
            cost_vectors=price_matrix,
            action_vectors=qty_matrix,
            user_id=uid,
        )

    return BehaviorPanel(
        _logs=logs,
        metadata={
            "dataset": "m5",
            "aggregation": "store",
            "goods": DEPARTMENTS,
            "min_weeks": min_weeks,
            "num_weeks_available": len(weeks_sorted),
        },
    )


def _build_store_dept_panel(
    sales_long,
    weeks_sorted: list[int],
    min_weeks: int,
    max_users: int | None,
) -> BehaviorPanel:
    """Build panel with one user per store-department, categories as goods."""
    import pandas as pd

    # Aggregate: total units and mean price per store-dept-cat-week
    agg = sales_long.groupby(
        ["store_id", "dept_id", "cat_id", "wm_yr_wk"],
    ).agg(
        units=("units", "sum"),
        price=("sell_price", "mean"),
    ).reset_index()

    combos = sorted(
        agg[["store_id", "dept_id"]].drop_duplicates().itertuples(index=False, name=None),
    )
    if max_users is not None:
        combos = combos[:max_users]

    logs: dict[str, BehaviorLog] = {}

    for store, dept in combos:
        subset = agg[(agg["store_id"] == store) & (agg["dept_id"] == dept)]

        # Pivot quantities: rows=weeks, columns=categories
        qty_pivot = subset.pivot_table(
            values="units", index="wm_yr_wk", columns="cat_id",
            aggfunc="sum",
        ).reindex(index=weeks_sorted, columns=CATEGORIES).fillna(0)

        # Pivot prices: rows=weeks, columns=categories
        price_pivot = subset.pivot_table(
            values="price", index="wm_yr_wk", columns="cat_id",
            aggfunc="mean",
        ).reindex(index=weeks_sorted, columns=CATEGORIES)
        price_pivot = price_pivot.ffill().bfill()

        active_mask = qty_pivot.sum(axis=1) > 0
        active_weeks = qty_pivot.index[active_mask].tolist()

        if len(active_weeks) < min_weeks:
            continue

        qty_matrix = qty_pivot.loc[active_weeks].values.astype(np.float64)
        price_matrix = price_pivot.loc[active_weeks].values.astype(np.float64)

        # Fill any remaining NaN prices with column median
        for col in range(price_matrix.shape[1]):
            mask = np.isnan(price_matrix[:, col])
            if mask.any():
                median_val = np.nanmedian(price_matrix[:, col])
                price_matrix[mask, col] = median_val if not np.isnan(median_val) else 1.0

        uid = f"{store}__{dept}"
        logs[uid] = BehaviorLog(
            cost_vectors=price_matrix,
            action_vectors=qty_matrix,
            user_id=uid,
        )

    return BehaviorPanel(
        _logs=logs,
        metadata={
            "dataset": "m5",
            "aggregation": "store_dept",
            "goods": CATEGORIES,
            "min_weeks": min_weeks,
            "num_weeks_available": len(weeks_sorted),
        },
    )
