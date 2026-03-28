#!/usr/bin/env python3
"""Generic budget RP variation diagnostics for supported datasets.

Datasets: open_ecommerce, hm

Examples:
  # Open E‑Commerce (Amazon)
  python tools/budget_variation.py --dataset open_ecommerce \
      --data-dir ~/.prefgraph/data/open_ecommerce --n-users 1000

  # H&M (heavier)
  python tools/budget_variation.py --dataset hm \
      --data-dir ~/.prefgraph/data/hm --max-users 2000 --top-k-groups 15
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


def _load_panel(dataset: str, args) -> tuple[any, list[str]]:
    if dataset == "open_ecommerce":
        from prefgraph.datasets import load_open_ecommerce
        panel = load_open_ecommerce(
            data_dir=args.data_dir,
            n_users=args.n_users,
            min_observations=args.min_observations,
            top_n_categories=args.top_n_categories,
        )
        goods = panel.metadata.get("goods", [])
        return panel, goods
    elif dataset == "hm":
        from prefgraph.datasets import load_hm
        panel = load_hm(
            data_dir=args.data_dir,
            max_users=args.max_users,
            min_periods=args.min_periods,
            top_k_groups=args.top_k_groups,
            time_period=args.time_period,
        )
        goods = panel.metadata.get("goods", panel.metadata.get("goods_labels", []))
        return panel, goods
    else:
        raise SystemExit(f"Unsupported dataset: {dataset}. Use open_ecommerce or hm.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["open_ecommerce", "hm"], required=True)
    ap.add_argument("--data-dir", type=str, default=None)

    # Open E‑Commerce args
    ap.add_argument("--n-users", type=int, default=None)
    ap.add_argument("--min-observations", type=int, default=5)
    ap.add_argument("--top-n-categories", type=int, default=50)

    # H&M args
    ap.add_argument("--max-users", type=int, default=50000)
    ap.add_argument("--min-periods", type=int, default=6)
    ap.add_argument("--top-k-groups", type=int, default=20)
    ap.add_argument("--time-period", choices=["week", "month", "quarter"], default="month")

    args = ap.parse_args()

    panel, goods = _load_panel(args.dataset, args)
    print(f"Loaded panel: users={panel.num_users}, entries={panel.num_entries}")
    print(f"Goods count: {len(goods)}")

    # Price variation diagnostics
    user_var_counts: list[int] = []
    user_T: list[int] = []
    user_price_cv_means: list[float] = []
    n_goods = len(goods)
    cat_values: list[set[float]] = [set() for _ in range(n_goods)]

    for _, log in panel:
        P = log.cost_vectors  # T x N
        T, N = P.shape
        user_T.append(T)
        distinct = np.unique(P, axis=0).shape[0]
        user_var_counts.append(int(distinct))

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
        print(f"  Median #unique monthly prices per category: {int(np.median(cat_unique_counts))}")
        print(f"  p75={int(np.percentile(cat_unique_counts, 75))}, p90={int(np.percentile(cat_unique_counts, 90))}")
        num_static_cats = sum(1 for k in cat_unique_counts if k <= 1)
        print(f"  Categories with no monthly price change: {num_static_cats}/{n_goods}")

    # RP analysis
    from prefgraph.engine import Engine
    engine = Engine(metrics=["garp", "ccei", "mpi"])  # auto-selects backend
    tuples = panel.to_engine_tuples()
    results = engine.analyze_arrays(tuples)

    garp_pass = float(np.mean([1.0 if r.is_garp else 0.0 for r in results]))
    ccei_vals = [float(r.ccei) for r in results]
    mpi_list = [float(r.mpi) for r in results if hasattr(r, "mpi")]

    ccei_p = _percentiles(ccei_vals, ps=(5, 25, 50, 75, 95))
    print("\nRP results summary:")
    print(f"  GARP pass rate: {_fmt_pct(garp_pass)}")
    print("  CCEI percentiles:", ", ".join(f"p{p}={ccei_p[p]:.3f}" for p in [5,25,50,75,95]))
    if mpi_list:
        mpi_p = _percentiles(mpi_list, ps=(5, 25, 50, 75, 95))
        print("  MPI percentiles:", ", ".join(f"p{p}={mpi_p[p]:.3f}" for p in [5,25,50,75,95]))

    # Variation vs CCEI
    aligned_var = []
    for _, log in panel:
        P = log.cost_vectors
        aligned_var.append(np.unique(P, axis=0).shape[0])
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
