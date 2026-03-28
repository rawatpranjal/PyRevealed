#!/usr/bin/env python3
"""Open E‑Commerce (Amazon) - Polars-based RP variation diagnostics.

Loads amazon-purchases.csv with Polars, prepares price/quantity panels
for ~N users, and runs GARP/CCEI/MPI to assess RP test power.

Usage:
  python tools/open_ecommerce_polars_variation.py \
    --data-dir case_studies/datasets/open_ecommerce/data \
    --n-users 5000 --min-observations 5 --top-n-categories 50
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import statistics
from collections import defaultdict


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _percentiles(arr: list[float], ps=(5, 25, 50, 75, 95)) -> dict[int, float]:
    if not arr:
        return {p: math.nan for p in ps}
    a = np.asarray(arr, dtype=float)
    return {p: float(np.nanpercentile(a, p)) for p in ps}


CATEGORY_KEYWORDS = [
    ("book", "Books"), ("kindle", "Books"),
    ("electronic", "Electronics"), ("computer", "Electronics"), ("phone", "Electronics"),
    ("clothing", "Clothing"), ("apparel", "Clothing"), ("shoe", "Clothing"),
    ("home", "Home & Garden"), ("garden", "Home & Garden"), ("kitchen", "Home & Garden"),
    ("grocery", "Grocery"), ("food", "Grocery"), ("gourmet", "Grocery"),
    ("health", "Health & Beauty"), ("beauty", "Health & Beauty"), ("personal care", "Health & Beauty"),
    ("toy", "Toys & Games"), ("game", "Toys & Games"),
    ("sport", "Sports & Outdoors"), ("outdoor", "Sports & Outdoors"),
    ("baby", "Baby Products"), ("pet", "Pet Supplies"), ("office", "Office Products"),
    ("automotive", "Automotive"), ("tool", "Tools & Home Improvement"),
    ("music", "Music & Entertainment"), ("movie", "Music & Entertainment"), ("video", "Music & Entertainment"),
]


def _map_category_py(cat: str) -> str:
    s = (cat or "").lower()
    for kw, grp in CATEGORY_KEYWORDS:
        if kw in s:
            return grp
    return "Other"


def build_panel_polars(data_dir: str, *, n_users: int | None, min_observations: int, top_n_categories: int):
    import polars as pl
    from prefgraph.core.session import BehaviorLog
    from prefgraph.core.panel import BehaviorPanel

    csv_path = f"{data_dir.rstrip('/')}/amazon-purchases.csv"

    # Lazy scan CSV
    scan = pl.scan_csv(csv_path, infer_schema_length=10_000)

    # Prepare base frame with period and category (raw Category values)
    df = (
        scan.select([
            pl.col("Survey ResponseID").alias("user"),
            pl.col("Order Date").str.strptime(pl.Datetime, strict=False).alias("order_dt"),
            pl.col("Category").alias("category"),
            pl.col("Purchase Price Per Unit").cast(pl.Float64).alias("price"),
            pl.col("Quantity").cast(pl.Float64).alias("qty"),
        ])
        .drop_nulls(subset=["order_dt"])
        .filter((pl.col("price") >= 0.01) & (pl.col("price") <= 1000.0) & (pl.col("qty") > 0))
        .with_columns([
            pl.col("order_dt").dt.strftime("%Y-%m").alias("period"),
            pl.when(pl.col("category").is_null()).then(pl.lit("Unknown")).otherwise(pl.col("category")).alias("category"),
        ])
        .select(["user", "period", "category", "price", "qty"])
    )

    # Top categories
    top_cats = (
        df.group_by("category").agg(pl.len().alias("cnt")).sort("cnt", descending=True).limit(top_n_categories)
        .collect().get_column("category").to_list()
    )
    df = df.filter(pl.col("category").is_in(top_cats))

    # Price oracle: median price per (period, category)
    oracle = df.group_by(["period", "category"]).agg(pl.median("price").alias("median_price")).collect()

    # Pivot oracle to grid (periods x categories), fill forward/backward, then global median
    periods_sorted = oracle.select("period").unique().sort("period").get_column("period").to_list()
    categories = sorted(top_cats)

    oracle_piv = (
        oracle.pivot(index="period", columns="category", values="median_price")
        .sort("period")
    )
    # Fill across time for each category column: forward then backward, then column median
    for c in categories:
        if c in oracle_piv.columns:
            oracle_piv = oracle_piv.with_columns(pl.col(c).fill_null(strategy="forward").alias(c))
            oracle_piv = oracle_piv.with_columns(pl.col(c).fill_null(strategy="backward").alias(c))
            col_median = float(oracle_piv.select(pl.col(c)).to_series().median())
            oracle_piv = oracle_piv.with_columns(pl.when(pl.col(c).is_null()).then(col_median).otherwise(pl.col(c)).alias(c))
        else:
            oracle_piv = oracle_piv.with_columns(pl.lit(float("nan")).alias(c))

    # Ensure all categories present and ordered
    oracle_piv = oracle_piv.select(["period", *categories])
    price_grid = oracle_piv.select(categories).to_numpy()
    period_to_idx = {p: i for i, p in enumerate(oracle_piv.get_column("period").to_list())}

    # Qualifying users (>= min_observations active periods)
    user_periods = (
        df.group_by(["user", "period"]).agg(pl.sum("qty").alias("qty_sum"))
        .filter(pl.col("qty_sum") > 0)
        .group_by("user").agg(pl.n_unique("period").alias("n_periods"))
        .filter(pl.col("n_periods") >= min_observations)
        .sort("n_periods", descending=True)
    ).collect()

    user_ids = user_periods.get_column("user").to_list()
    if n_users is not None:
        user_ids = user_ids[:n_users]

    # Materialize filtered data for selected users
    users_df = df.filter(pl.col("user").is_in(user_ids)).collect()

    logs = {}
    for uid, user_df in users_df.partition_by("user", as_dict=True).items():
        # Quantity pivot (period x category)
        qty_piv = (
            user_df.pivot(index="period", columns="category", values="qty", aggregate_function="sum")
            .with_columns([pl.col(c).fill_null(0.0).alias(c) for c in user_df.get_column("category").unique().to_list()])
        )
        # Ensure all categories present and ordered
        for c in categories:
            if c not in qty_piv.columns:
                qty_piv = qty_piv.with_columns(pl.lit(0.0).alias(c))
        qty_piv = qty_piv.select(["period", *categories]).sort("period")

        # Active periods
        qty_mat = qty_piv.select(categories).to_numpy()
        row_sums = np.sum(qty_mat, axis=1)
        active_mask = row_sums > 0
        if not np.any(active_mask):
            continue
        active_periods = [p for p, m in zip(qty_piv.get_column("period").to_list(), active_mask) if m]
        if len(active_periods) < min_observations:
            continue

        qty_matrix = qty_piv.filter(pl.Series(active_mask)).select(categories).to_numpy().astype(float)

        # User realized prices pivot (median per cell)
        price_piv = (
            user_df.pivot(index="period", columns="category", values="price", aggregate_function="median")
        )
        for c in categories:
            if c not in price_piv.columns:
                price_piv = price_piv.with_columns(pl.lit(float("nan")).alias(c))
        price_piv = price_piv.select(["period", *categories]).sort("period")

        # Align to oracle grid indices
        price_idx = []
        for p in active_periods:
            if p in period_to_idx:
                price_idx.append(period_to_idx[p])
            else:
                price_idx.append(None)
        # Drop periods missing from oracle
        valid = [i for i, v in enumerate(price_idx) if v is not None]
        if len(valid) != len(price_idx):
            active_periods = [active_periods[i] for i in valid]
            qty_matrix = qty_matrix[valid, :]
            price_idx = [price_idx[i] for i in valid]
            if len(active_periods) < min_observations:
                continue

        user_price_matrix = price_piv.filter(pl.col("period").is_in(active_periods)).select(categories).to_numpy().astype(float)
        market_slice = price_grid[np.array(price_idx, dtype=int), :].astype(float)
        price_matrix = np.where(np.isnan(user_price_matrix), market_slice, user_price_matrix)

        logs[str(uid)] = BehaviorLog(cost_vectors=price_matrix, action_vectors=qty_matrix, user_id=str(uid))

    return BehaviorPanel(_logs=logs, metadata={"dataset": "open_ecommerce", "goods": categories})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True, help="Directory containing amazon-purchases.csv")
    ap.add_argument("--n-users", type=int, default=5000)
    ap.add_argument("--min-observations", type=int, default=5)
    ap.add_argument("--top-n-categories", type=int, default=50)
    args = ap.parse_args()

    panel = build_panel_polars(
        args.data_dir,
        n_users=args.n_users,
        min_observations=args.min_observations,
        top_n_categories=args.top_n_categories,
    )
    print(f"Built panel (Polars path): users={panel.num_users}, entries={panel.num_entries}, goods={len(panel.metadata.get('goods', []))}")

    # Variation diagnostics
    user_var_counts: list[int] = []
    user_T: list[int] = []
    user_price_cv_means: list[float] = []
    n_goods = len(panel.metadata.get("goods", []))
    cat_values: list[set[float]] = [set() for _ in range(n_goods)]

    for _, log in panel:
        P = log.cost_vectors
        T, N = P.shape
        user_T.append(T)
        user_var_counts.append(int(np.unique(P, axis=0).shape[0]))
        cvs = []
        for j in range(N):
            col = P[:, j]
            m = float(np.mean(col))
            s = float(np.std(col))
            if m > 0:
                cvs.append(s / m)
            for v in col:
                cat_values[j].add(float(v))
        if cvs:
            user_price_cv_means.append(float(np.mean(cvs)))

    print("\nPrice variation (user-level):")
    print(f"  Users with >1 distinct price vector: {_fmt_pct(sum(1 for x in user_var_counts if x > 1)/max(1,len(user_var_counts)))}")
    print(f"  Users with ≥3 distinct price vectors: {_fmt_pct(sum(1 for x in user_var_counts if x >= 3)/max(1,len(user_var_counts)))}")
    print(f"  Median observations per user (T): {int(np.median(user_T)) if user_T else 0}")
    cvp = _percentiles(user_price_cv_means)
    print("  Mean CV of prices across categories per user (percentiles):")
    print("   ", ", ".join(f"p{p}={cvp[p]:.3f}" for p in [5,25,50,75,95]))

    # RP tests
    from prefgraph.engine import Engine
    engine = Engine(metrics=["garp", "ccei", "mpi"])  # auto backend
    results = engine.analyze_arrays(panel.to_engine_tuples())
    garp_pass = float(np.mean([1.0 if r.is_garp else 0.0 for r in results]))
    ccei_vals = [float(r.ccei) for r in results]
    mpi_vals = [float(r.mpi) for r in results]
    ccei_p = _percentiles(ccei_vals)

    print("\nRP results summary:")
    print(f"  GARP pass rate: {_fmt_pct(garp_pass)}")
    print("  CCEI percentiles:", ", ".join(f"p{p}={ccei_p[p]:.3f}" for p in [5,25,50,75,95]))

    # Variation vs CCEI
    aligned_var = []
    for _, log in panel:
        aligned_var.append(int(np.unique(log.cost_vectors, axis=0).shape[0]))
    if len(aligned_var) == len(ccei_vals) and len(ccei_vals) > 0:
        try:
            corr = float(np.corrcoef(np.asarray(aligned_var, float), np.asarray(ccei_vals, float))[0, 1])
        except Exception:
            corr = float("nan")
        print("\nVariation vs CCEI:")
        print(f"  Corr(#distinct price vectors, CCEI): {corr:.3f}")
        bins = defaultdict(list)
        for v, c in zip(aligned_var, ccei_vals):
            bins[min(int(v), 5)].append(float(c))
        for k in sorted(bins):
            try:
                mean_c = statistics.mean(bins[k])
            except statistics.StatisticsError:
                mean_c = float("nan")
            print(f"   distinct={k:>2}: n={len(bins[k]):>5}, mean CCEI={mean_c:.3f}")


if __name__ == "__main__":
    main()
