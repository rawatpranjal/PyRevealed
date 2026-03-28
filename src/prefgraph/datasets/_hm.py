"""H&M Fashion dataset loader.

Loads the H&M Personalized Fashion Recommendations dataset of ~1.36M
customers purchasing clothing articles over 2 years (2018-09 to 2020-09),
returning a BehaviorPanel.

Articles are aggregated into product groups (first 2 digits of article_id).
Transactions are aggregated to configurable time periods (week/month/quarter).

Price construction uses per-customer realized prices:
  - Purchased groups: customer's average paid price in that period-group
  - Unpurchased groups: period-group median -> group median -> global median

Data must be downloaded separately from Kaggle.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from prefgraph.core.panel import BehaviorPanel
from prefgraph.core.session import BehaviorLog

# --- Constants ---

MAX_PRODUCT_GROUPS = 20
DEFAULT_MAX_USERS = 50_000
DEFAULT_MIN_PERIODS = 6
CHUNKSIZE = 500_000
CUTOFF_DATE = "2020-06-01"

VALID_PERIODS = {"week": "W", "month": "M", "quarter": "Q"}


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find H&M data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "hm")

    candidates.extend([
        Path.home() / ".prefgraph" / "data" / "hm",
        Path(__file__).resolve().parents[3] / "hm" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (d / "transactions_train.csv").exists():
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"H&M data not found. Searched:\n  {searched}\n\n"
        "Download from Kaggle: https://www.kaggle.com/competitions/"
        "h-and-m-personalized-fashion-recommendations/data\n"
        "Place transactions_train.csv in the data directory.\n"
        "Then pass data_dir= or set PYREVEALED_DATA_DIR environment variable."
    )


def load_hm(
    data_dir: str | Path | None = None,
    max_users: int = DEFAULT_MAX_USERS,
    min_periods: int = DEFAULT_MIN_PERIODS,
    top_k_groups: int = MAX_PRODUCT_GROUPS,
    cutoff_date: str = CUTOFF_DATE,
    time_period: str = "month",
) -> BehaviorPanel:
    """Load H&M Fashion dataset as a BehaviorPanel.

    Reads transactions_train.csv in chunks for memory efficiency.
    Maps article_id to product groups (first 2 digits), aggregates
    to per-customer price-quantity panels.

    Price construction: for purchased groups, the customer's own average
    realized price is used. For unpurchased groups, prices are imputed
    via period-group median -> group median -> global median fallback.

    Args:
        data_dir: Path to directory containing transactions_train.csv.
        max_users: Maximum number of customers (most active, default 50000).
        min_periods: Minimum active periods per customer (default 6).
        top_k_groups: Number of top product groups to keep (default 20).
        cutoff_date: ISO date for metadata (default '2020-06-01').
        time_period: Aggregation period - "week", "month" (default),
            or "quarter".

    Returns:
        BehaviorPanel with one BehaviorLog per customer.

    Raises:
        FileNotFoundError: If data files cannot be found.
        ImportError: If pandas is not installed.
        ValueError: If time_period is invalid.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for dataset loaders. "
            "Install with: pip install 'prefgraph[datasets]'"
        ) from None

    if time_period not in VALID_PERIODS:
        raise ValueError(
            f"time_period must be one of {list(VALID_PERIODS)}, got {time_period!r}"
        )
    pd_freq = VALID_PERIODS[time_period]

    data_path = _find_data_dir(data_dir)
    csv_path = data_path / "transactions_train.csv"

    # --- Pass 1: chunked scan to find top product groups and active users ---
    # Two-pass design is necessary: 3.49 GB CSV cannot fit in memory.
    # Pass 1 reads only customer_id + article_id (no prices) to identify
    # the top-K product groups and most-active users before loading prices.
    group_counts: dict[str, int] = {}
    user_counts: dict[str, int] = {}

    for chunk in pd.read_csv(
        csv_path,
        usecols=["customer_id", "article_id"],
        dtype={"customer_id": str, "article_id": str},
        chunksize=CHUNKSIZE,
    ):
        # Product group = first 2 digits of article_id. This is the coarsest
        # grouping available without the articles.csv metadata file. Produces
        # ~20 groups with repeated support across months - essential for RP.
        chunk["product_group"] = chunk["article_id"].str[:2]

        for grp, cnt in chunk["product_group"].value_counts().items():
            group_counts[grp] = group_counts.get(grp, 0) + cnt

        for uid, cnt in chunk["customer_id"].value_counts().items():
            user_counts[uid] = user_counts.get(uid, 0) + cnt

    sorted_groups = sorted(group_counts, key=group_counts.get, reverse=True)
    top_groups = sorted_groups[:top_k_groups]

    sorted_users = sorted(user_counts, key=user_counts.get, reverse=True)
    target_users = set(sorted_users[:max_users])

    # --- Pass 2: chunked load of filtered data ---
    frames = []

    for chunk in pd.read_csv(
        csv_path,
        dtype={"customer_id": str, "article_id": str, "sales_channel_id": int},
        parse_dates=["t_dat"],
        chunksize=CHUNKSIZE,
    ):
        chunk["product_group"] = chunk["article_id"].str[:2]
        mask = (
            chunk["customer_id"].isin(target_users)
            & chunk["product_group"].isin(top_groups)
        )
        if mask.any():
            frames.append(chunk.loc[mask, [
                "t_dat", "customer_id", "product_group", "price",
            ]])

    df = pd.concat(frames, ignore_index=True)

    # --- Period key ---
    df["period"] = df["t_dat"].dt.to_period(pd_freq)
    periods_sorted = sorted(df["period"].unique())
    period_to_idx = {p: i for i, p in enumerate(periods_sorted)}
    period_labels = [str(p) for p in periods_sorted]

    # --- Three-tier imputation oracle ---
    # RP tests require a FULL price vector every period (purchased + unpurchased
    # groups). For purchased groups we use the customer's own realized price.
    # For unpurchased groups we need an imputation. The fallback chain is:
    #   1. period-group median (most specific - "what did others pay this month?")
    #   2. group median (across all periods - "what does this group typically cost?")
    #   3. global median (last resort - "what does anything cost?")
    # The old loader used a single shared median oracle for ALL customers,
    # which destroyed individual price variation. Per-customer prices let RP
    # detect when a customer paid more/less than the market for a group.
    period_group_median = df.groupby(["period", "product_group"])["price"].median()
    group_median = df.groupby("product_group")["price"].median()
    global_median = float(df["price"].median())

    # Build (n_periods, n_groups) grid, filling from broadest to most specific
    # so that more specific values overwrite broader ones.
    impute_grid = np.full((len(periods_sorted), len(top_groups)), global_median)
    for gi, grp in enumerate(top_groups):
        if grp in group_median.index:
            impute_grid[:, gi] = group_median[grp]
        for pi, per in enumerate(periods_sorted):
            if (per, grp) in period_group_median.index:
                impute_grid[pi, gi] = period_group_median[(per, grp)]

    # --- Aggregate: quantity (count) + realized mean price per customer-period-group ---
    # Quantity = number of article rows in the cell. The raw H&M data has duplicate
    # (date, customer, article) rows which represent distinct purchased units, so
    # row counts are valid quantities, not transaction counts.
    # mean_price = customer's own average paid price for that group in that period.
    agg = df.groupby(["customer_id", "period", "product_group"]).agg(
        quantity=("price", "size"),
        mean_price=("price", "mean"),
    ).reset_index()

    # --- Build per-customer BehaviorLogs ---
    # This loop is the bottleneck at scale: two pivot_table calls per user.
    # At 50K users, takes ~10 min (vs ~2 min with the old shared-oracle approach).
    # Vectorizing would require a 3D sparse tensor which pandas doesn't support.
    logs: dict[str, BehaviorLog] = {}

    for cid, cust_data in agg.groupby("customer_id"):
        # Pivot quantity
        qty_pivot = cust_data.pivot_table(
            values="quantity",
            index="period",
            columns="product_group",
            aggfunc="sum",
        ).reindex(index=periods_sorted, columns=top_groups).fillna(0)

        # Pivot realized prices (NaN where customer didn't purchase)
        price_pivot = cust_data.pivot_table(
            values="mean_price",
            index="period",
            columns="product_group",
            aggfunc="mean",
        ).reindex(index=periods_sorted, columns=top_groups)

        # Active periods: at least one purchase in any group
        active_mask = qty_pivot.sum(axis=1) > 0
        active_periods = qty_pivot.index[active_mask].tolist()

        if len(active_periods) < min_periods:
            continue

        active_indices = [period_to_idx[p] for p in active_periods]

        qty_matrix = qty_pivot.loc[active_periods].values.astype(np.float64)

        # Per-customer prices: realized where purchased, imputed where not.
        # price_raw has NaN exactly where qty == 0 (reindex produced NaN for
        # groups the customer didn't buy). np.where swaps those NaNs for the
        # imputation grid values while keeping realized prices intact.
        price_raw = price_pivot.loc[active_periods].values.astype(np.float64)
        impute_slice = impute_grid[active_indices]
        price_matrix = np.where(np.isnan(price_raw), impute_slice, price_raw)

        uid = f"customer_{cid[:12]}"
        logs[uid] = BehaviorLog(
            cost_vectors=price_matrix,
            action_vectors=qty_matrix,
            user_id=uid,
        )

    return BehaviorPanel(
        _logs=logs,
        metadata={
            "dataset": "hm",
            "goods": top_groups,
            "goods_labels": [f"group_{g}" for g in top_groups],
            "periods": period_labels,
            "time_period": time_period,
            "min_periods": min_periods,
            "max_users": max_users,
            "top_k_groups": top_k_groups,
            "cutoff_date": cutoff_date,
            "num_periods_available": len(periods_sorted),
        },
    )
