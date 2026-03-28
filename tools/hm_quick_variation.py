#!/usr/bin/env python3
"""H&M quick RP variation (sampled rows).

Reads a sample of transactions (nrows) to approximate price variation and
run RP tests without scanning the full 3.5GB file.

Usage:
  python tools/hm_quick_variation.py \
    --data-dir /Volumes/Expansion/pyrevealed_data/hm \
    --nrows 500000 --top-k-groups 15 --max-users 2000 --min-periods 6 \
    --time-period month
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict

import numpy as np


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _percentiles(arr: list[float], ps=(5, 25, 50, 75, 95)) -> dict[int, float]:
    if not arr:
        return {p: math.nan for p in ps}
    a = np.asarray(arr, dtype=float)
    return {p: float(np.nanpercentile(a, p)) for p in ps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--nrows", type=int, default=500_000, help="Rows to read from CSV")
    ap.add_argument("--top-k-groups", type=int, default=15)
    ap.add_argument("--max-users", type=int, default=2000)
    ap.add_argument("--min-periods", type=int, default=6)
    ap.add_argument("--time-period", choices=["week", "month", "quarter"], default="month")
    args = ap.parse_args()

    import pandas as pd
    from pathlib import Path

    csv_path = Path(args.data_dir) / "transactions_train.csv"
    if not csv_path.exists():
        raise SystemExit(f"transactions_train.csv not found in {args.data_dir}")

    usecols = ["customer_id", "article_id", "price", "t_dat"]
    print(f"Reading sample: {args.nrows:,} rows from {csv_path}")
    df = pd.read_csv(csv_path, usecols=usecols, nrows=args.nrows, parse_dates=["t_dat"], dtype={"customer_id": str, "article_id": str})

    # Coarse product group and period
    df["product_group"] = df["article_id"].str[:2]
    freq = {"week": "W", "month": "M", "quarter": "Q"}[args.time_period]
    df["period"] = df["t_dat"].dt.to_period(freq)

    # Top groups by count (within sample)
    top_groups = df["product_group"].value_counts().head(args.top_k_groups).index.tolist()
    df = df[df["product_group"].isin(top_groups)].copy()

    # Top users by activity (within sample)
    top_users = df["customer_id"].value_counts().head(args.max_users * 2).index.tolist()
    df = df[df["customer_id"].isin(top_users)].copy()

    # Build global imputation oracle (period×group median)
    period_group_median = df.groupby(["period", "product_group"])['price'].median()
    periods_sorted = sorted(df["period"].unique())
    groups_sorted = top_groups
    impute_grid = np.full((len(periods_sorted), len(groups_sorted)), float(df['price'].median()))
    per_to_idx = {p: i for i, p in enumerate(periods_sorted)}
    grp_to_idx = {g: j for j, g in enumerate(groups_sorted)}
    for (per, grp), val in period_group_median.items():
        if per in per_to_idx and grp in grp_to_idx:
            impute_grid[per_to_idx[per], grp_to_idx[grp]] = float(val)

    # Aggregate per user-period-group
    agg = df.groupby(["customer_id", "period", "product_group"]).agg(
        quantity=("price", "size"),
        mean_price=("price", "mean"),
    ).reset_index()

    from prefgraph.core.session import BehaviorLog
    from prefgraph.core.panel import BehaviorPanel

    logs = {}
    kept = 0
    for cid, cust in agg.groupby("customer_id"):
        # Pivot quantity and price
        qty_pivot = cust.pivot_table(values="quantity", index="period", columns="product_group", aggfunc="sum").reindex(columns=groups_sorted).fillna(0)
        price_pivot = cust.pivot_table(values="mean_price", index="period", columns="product_group", aggfunc="mean").reindex(columns=groups_sorted)

        active_mask = qty_pivot.sum(axis=1) > 0
        active_periods = qty_pivot.index[active_mask].tolist()
        if len(active_periods) < args.min_periods:
            continue

        # Align to impute grid
        active_idx = [per_to_idx[p] for p in active_periods if p in per_to_idx]
        if len(active_idx) != len(active_periods):
            continue

        qty = qty_pivot.loc[active_periods].values.astype(float)
        realized = price_pivot.loc[active_periods].values.astype(float)
        impute_slice = impute_grid[active_idx]
        prices = np.where(np.isnan(realized), impute_slice, realized)

        logs[cid] = BehaviorLog(cost_vectors=prices, action_vectors=qty, user_id=f"customer_{cid[:12]}")
        kept += 1
        if kept >= args.max_users:
            break

    panel = BehaviorPanel(_logs=logs, metadata={"dataset": "hm_quick", "goods": groups_sorted})
    print(f"Built panel: users={panel.num_users}, entries={panel.num_entries}")

    # Variation diagnostics
    user_var_counts = []
    user_T = []
    user_price_cv_means = []
    cat_values = [set() for _ in range(len(groups_sorted))]
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

    pct_users_mult_prices = sum(1 for x in user_var_counts if x > 1) / max(1, len(user_var_counts))
    pct_users_many_prices = sum(1 for x in user_var_counts if x >= 3) / max(1, len(user_var_counts))
    median_T = int(np.median(user_T)) if user_T else 0
    cv_percentiles = _percentiles(user_price_cv_means)
    print("\nPrice variation (user-level):")
    print(f"  Users with >1 distinct price vector: {_fmt_pct(pct_users_mult_prices)}")
    print(f"  Users with ≥3 distinct price vectors: {_fmt_pct(pct_users_many_prices)}")
    print(f"  Median observations per user (T): {median_T}")
    print("  Mean CV of prices across categories per user (percentiles):")
    print("   ", ", ".join(f"p{p}={cv_percentiles[p]:.3f}" for p in [5,25,50,75,95]))

    cat_unique_counts = [len(s) for s in cat_values]
    if cat_unique_counts:
        print("\nPrice variation (category-level):")
        print(f"  Median #unique prices per category: {int(np.median(cat_unique_counts))}")
        print(f"  p75={int(np.percentile(cat_unique_counts, 75))}, p90={int(np.percentile(cat_unique_counts, 90))}")
        num_static_cats = sum(1 for k in cat_unique_counts if k <= 1)
        print(f"  Categories with no price change: {num_static_cats}/{len(groups_sorted)}")

    # RP tests
    from prefgraph.engine import Engine
    engine = Engine(metrics=["garp", "ccei", "mpi"])  # auto backend
    results = engine.analyze_arrays(panel.to_engine_tuples())
    garp_pass = float(np.mean([1.0 if r.is_garp else 0.0 for r in results]))
    ccei_vals = [float(r.ccei) for r in results]
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
            corr = np.corrcoef(np.asarray(aligned_var, float), np.asarray(ccei_vals, float))[0, 1]
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
