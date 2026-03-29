#!/usr/bin/env python3
"""ML Benchmark Runner: Revealed preference features as predictive signals.

Runs three-way comparison (RP-only vs Baseline-only vs Combined) across
multiple real-world datasets and prediction targets.

Usage:
    python case_studies/benchmarks/runner.py                     # All available datasets
    python case_studies/benchmarks/runner.py --datasets dunnhumby,uci_retail
    python case_studies/benchmarks/runner.py --datasets dunnhumby --max-users 500
    python case_studies/benchmarks/runner.py --skip-existing      # Skip datasets with cached results
    python case_studies/benchmarks/runner.py --replot             # Regenerate summary + plots from cache
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from case_studies.benchmarks.core.evaluation import BenchmarkResult
from case_studies.benchmarks.core.reporting import (
    print_summary,
    save_results,
    save_dataset_results,
    load_dataset_results,
    aggregate_all_results,
    generate_plots,
)


# Registry: name -> module path
# To add a new dataset: create datasets/<name>_bench.py with run_benchmark() -> list[BenchmarkResult],
# then add an entry here.
AVAILABLE_DATASETS = {
    # Budget-based (real prices)
    "dunnhumby": "case_studies.benchmarks.datasets.dunnhumby_bench",
    "open_ecommerce": "case_studies.benchmarks.datasets.open_ecommerce_bench",
    "hm": "case_studies.benchmarks.datasets.hm_bench",
    # Menu-based
    "instacart_v2_menu": "case_studies.benchmarks.datasets.instacart_v2_menu_bench",
    "rees46": "case_studies.benchmarks.datasets.rees46_bench",
    "taobao": "case_studies.benchmarks.datasets.taobao_bench",
    "taobao_buy_window": "case_studies.benchmarks.datasets.taobao_buy_window_bench",
    "retailrocket": "case_studies.benchmarks.datasets.retailrocket_bench",
    "tenrec": "case_studies.benchmarks.datasets.tenrec_bench",
    "yoochoose": "case_studies.benchmarks.datasets.yoochoose_bench",
    "kuairec": "case_studies.benchmarks.datasets.kuairec_bench",
    "mind": "case_studies.benchmarks.datasets.mind_bench",
    "finn_slates": "case_studies.benchmarks.datasets.finn_slates_bench",
}

# Map runner name -> display name used in BenchmarkResult.dataset field.
# Used by --skip-existing to find cached results.
DATASET_DISPLAY_NAMES = {
    "dunnhumby": "Dunnhumby",
    "open_ecommerce": "Open E-Commerce",
    "hm": "H&M",
    "instacart_v2_menu": "Instacart V2 (Menu)",
    "rees46": "REES46",
    "taobao": "Taobao",
    "taobao_buy_window": "Taobao (Buy Window)",
    "retailrocket": "RetailRocket",
    "tenrec": "Tenrec",
    "yoochoose": "Yoochoose",
    "kuairec": "KuaiRec",
    "mind": "MIND",
    "finn_slates": "FINN.no Slates",
}

# Datasets validated for RP feature computation (see datasets_issues.md).
# Excluded: kuairec (post-hoc choice assignment, not revealed preference),
#           yoochoose (synthetic users, violates same-agent assumption).
VALIDATED_DATASETS = [
    "dunnhumby", "open_ecommerce", "hm",
    "instacart_v2_menu", "rees46", "taobao", "taobao_buy_window",
    "retailrocket", "tenrec",
    "mind", "finn_slates",
]


def run_dataset(
    name: str,
    max_users: int | None = None,
    *,
    taobao_window_seconds: int | None = None,
    n_rows: int | None = None,
) -> list[BenchmarkResult]:
    """Run benchmark for a single dataset."""
    import importlib

    module_path = AVAILABLE_DATASETS[name]
    mod = importlib.import_module(module_path)

    kwargs = {}
    if name == "dunnhumby":
        if max_users:
            kwargs["n_households"] = max_users
    elif name == "open_ecommerce":
        if max_users:
            kwargs["n_users"] = max_users
    elif name == "mind":
        # MIND's 1-click filter is aggressive: 250 raw users → ~23 qualifying.
        # Inflate the cap so enough users survive filtering.
        effective = max_users or 50000
        if max_users is not None and max_users < 2000:
            effective = max(max_users * 20, 5000)
        kwargs["max_users"] = effective
    elif name in ("instacart_v2_menu", "rees46", "hm", "taobao", "retailrocket", "tenrec", "yoochoose", "finn_slates"):
        kwargs["max_users"] = max_users or 50000
    elif name == "kuairec":
        # KuaiRec has only 1411 users; max_users=None means all users
        if max_users:
            kwargs["max_users"] = max_users
    elif name == "taobao_buy_window":
        kwargs["max_users"] = max_users or 50000
        if taobao_window_seconds is not None:
            kwargs["window_seconds"] = taobao_window_seconds
        if n_rows is not None:
            kwargs["n_rows"] = n_rows

    try:
        return mod.run_benchmark(**kwargs)
    except FileNotFoundError as e:
        print(f"\n  [SKIP] {name}: {e}")
        return []
    except Exception as e:
        print(f"\n  [ERROR] {name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(description="ML Benchmark: RP Features as Predictive Signals")
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=f"Comma-separated names, 'all', or 'validated'. Available: {', '.join(AVAILABLE_DATASETS)}",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Cap number of users per dataset (for quick testing).",
    )
    parser.add_argument(
        "--taobao-window",
        type=int,
        default=None,
        help="Buy-anchored window size in seconds (only for taobao_buy_window)",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="Rows to read from CSV for datasets that support partial loads (e.g., taobao_buy_window)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: case_studies/benchmarks/output/)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets that already have cached results in the output directory.",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Load all cached results and regenerate summary + plots without running benchmarks.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        choices=["lgbm", "lasso", "both"],
        help="Model to run: lgbm (default), lasso, or both.",
    )
    parser.add_argument(
        "--eda-only",
        action="store_true",
        help="Load datasets and compute EDA summary only — no ML models.",
    )
    args = parser.parse_args()

    if args.datasets == "all":
        dataset_names = list(AVAILABLE_DATASETS.keys())
    elif args.datasets == "validated":
        dataset_names = VALIDATED_DATASETS
    else:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
        for d in dataset_names:
            if d not in AVAILABLE_DATASETS:
                print(f"Unknown dataset: {d}. Available: {', '.join(AVAILABLE_DATASETS)}")
                sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "output"

    # --replot: just load cached results and regenerate outputs
    if args.replot:
        print("=" * 70)
        print(" REPLOT: Loading cached results and regenerating outputs")
        print("=" * 70)
        all_results = aggregate_all_results(output_dir)
        if all_results:
            print(f"\n  Loaded {len(all_results)} cached results from {output_dir}")
            print_summary(all_results)
            save_results(all_results, output_dir)
            generate_plots(all_results, output_dir)
        else:
            print(f"\n  No cached results found in {output_dir}")
        return

    # --eda-only: load datasets, compute EDA from raw data, no ML
    if args.eda_only:
        import importlib as _imp
        from case_studies.benchmarks.core.eda import (
            compute_correlation_summary, print_correlation_summary,
            print_eda_summary, save_eda,
        )
        print("=" * 70)
        print(" EDA ONLY: Loading datasets, computing EDA + feature correlations")
        print("=" * 70)

        eda_list = []
        corr_list = []

        for name in _iter(dataset_names, "EDA"):
            display = DATASET_DISPLAY_NAMES.get(name, name)
            try:
                # Call load_and_prepare directly (skips ML models)
                mod_path = AVAILABLE_DATASETS[name]
                mod = _imp.import_module(mod_path)

                # Build kwargs (same logic as run_dataset)
                kwargs = {}
                if name == "dunnhumby":
                    if args.max_users:
                        kwargs["n_households"] = args.max_users
                elif name == "open_ecommerce":
                    if args.max_users:
                        kwargs["n_users"] = args.max_users
                elif name == "mind":
                    effective = args.max_users or 50000
                    if args.max_users and args.max_users < 2000:
                        effective = max(args.max_users * 20, 5000)
                    kwargs["max_users"] = effective
                elif name == "kuairec":
                    if args.max_users:
                        kwargs["max_users"] = args.max_users
                elif name == "taobao_buy_window":
                    kwargs["max_users"] = args.max_users or 50000
                else:
                    kwargs["max_users"] = args.max_users or 50000

                raw = mod.load_and_prepare(**kwargs)
                if len(raw) == 5:
                    X_rp, X_base, _, targets_dict, user_ids = raw
                else:
                    X_rp, X_base, targets_dict, user_ids = raw

                if X_rp is None:
                    print(f"  [{name}] Too few users, skipping")
                    continue

                # EDA (stored by load_and_prepare)
                eda = getattr(mod.load_and_prepare, "eda", None)
                if eda is not None:
                    eda["dataset"] = display
                    eda_list.append(eda)

                # Feature correlations
                corr = compute_correlation_summary(X_rp, X_base)
                corr["dataset"] = display
                corr_list.append(corr)
                print_correlation_summary(corr, display)

            except FileNotFoundError as e:
                print(f"  [SKIP] {name}: {e}")
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
                import traceback
                traceback.print_exc()

        if eda_list:
            print_eda_summary(eda_list)
            # Save EDA + correlations together
            for eda, corr in zip(eda_list, corr_list):
                eda["correlation"] = corr
            save_eda(eda_list, output_dir)
        return

    run_lgbm = args.model in ("lgbm", "both")
    run_lasso = args.model in ("lasso", "both")

    print("=" * 70)
    print(" ML BENCHMARK: Revealed Preference Features as Predictive Signals")
    print("=" * 70)
    print(f"\n  Datasets: {', '.join(dataset_names)}")
    print(f"  Max users: {args.max_users or 'unlimited'}")
    print(f"  Model(s): {args.model}")
    print(f"  Output: {output_dir}")
    if args.skip_existing:
        print(f"  Mode: skip-existing (use cached results where available)")

    all_results: list[BenchmarkResult] = []
    start = time.time()

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    def _iter(names, desc):
        if tqdm is not None:
            return tqdm(names, desc=desc, unit="ds")
        return names

    # ---- LightGBM ----
    if run_lgbm:
        print("\n" + "-" * 50)
        print(" LightGBM")
        print("-" * 50)
        for name in _iter(dataset_names, "LGBM"):
            display_name = DATASET_DISPLAY_NAMES.get(name, name)

            if args.skip_existing:
                cached = load_dataset_results(display_name, output_dir)
                if cached:
                    print(f"\n  [{name}] Using cached results ({len(cached)} targets)")
                    all_results.extend(cached)
                    continue

            t0 = time.time()
            results = run_dataset(
                name,
                args.max_users,
                taobao_window_seconds=args.taobao_window,
                n_rows=args.n_rows,
            )
            elapsed = time.time() - t0
            print(f"  [{name}] Completed in {elapsed:.1f}s ({len(results)} targets)")

            if results:
                save_dataset_results(results, results[0].dataset, output_dir)

            all_results.extend(results)

    # ---- Logit-Lasso ----
    if run_lasso:
        print("\n" + "-" * 50)
        print(" Logit-Lasso / LassoCV")
        print("-" * 50)
        from case_studies.benchmarks.lasso_benchmark import (
            DATASETS as LASSO_DATASETS,
            run_lasso_three_way,
            load_dataset as lasso_load_dataset,
            print_fit_table,
            print_three_way_table,
            print_coefficient_table,
            save_lasso_results,
            LassoResult,
        )

        lasso_results: list[LassoResult] = []
        for name in _iter(dataset_names, "Lasso"):
            if name not in LASSO_DATASETS:
                print(f"  [{name}] Not in lasso registry, skipping")
                continue

            try:
                raw = lasso_load_dataset(name, args.max_users)
                if len(raw) == 5:
                    X_rp, X_base, _, targets_dict, user_ids = raw
                else:
                    X_rp, X_base, targets_dict, user_ids = raw
            except FileNotFoundError as e:
                print(f"\n  [SKIP] {name}: {e}")
                continue
            except Exception as e:
                print(f"\n  [ERROR] {name}: {e}")
                import traceback
                traceback.print_exc()
                continue

            if X_rp is None:
                print(f"\n  [SKIP] {name}: too few users")
                continue

            for target_name, target_tuple in targets_dict.items():
                if len(target_tuple) == 4:
                    y, task_type, y_cont, pctl = target_tuple
                    if y_cont is not None:
                        y_cont = np.asarray(y_cont)
                else:
                    y, task_type = target_tuple
                    y_cont, pctl = None, None

                if task_type == "classification":
                    pr = float(np.mean(y))
                    if pr < 0.02 or pr > 0.98:
                        print(f"  [{name}] Skipping {target_name} — too imbalanced ({pr:.3f})")
                        continue

                print(f"  [{name}] {target_name} ({task_type})...", end=" ", flush=True)
                try:
                    result = run_lasso_three_way(
                        X_rp, X_base, y, name, target_name, task_type,
                        y_continuous=y_cont, threshold_pctl=pctl,
                    )
                except Exception as e:
                    print(f"SKIP ({e})")
                    continue

                lasso_results.append(result)
                if task_type == "classification":
                    gap = result.auc_pr_combined_train - result.auc_pr_combined
                    print(f"AP={result.auc_pr_combined:.3f} (train={result.auc_pr_combined_train:.3f} gap={gap:+.2f})  "
                          f"{result.n_features_nonzero} feat  [{result.wall_time_s:.1f}s]")
                else:
                    gap = result.r2_combined_train - result.r2_combined
                    print(f"R²={result.r2_combined:.3f} (train={result.r2_combined_train:.3f} gap={gap:+.2f})  "
                          f"{result.n_features_nonzero} feat  [{result.wall_time_s:.1f}s]")

        if lasso_results:
            print_fit_table(lasso_results)
            print_three_way_table(lasso_results)
            print_coefficient_table(lasso_results)
            save_lasso_results(lasso_results, output_dir)

    total_time = time.time() - start
    print(f"\nTotal time: {total_time:.1f}s")

    if all_results:
        print_summary(all_results)
        save_results(all_results, output_dir)
        generate_plots(all_results, output_dir)
    elif not run_lgbm:
        pass  # lasso-only mode, results already printed
    else:
        print("\nNo results - check that at least one dataset is available.")


if __name__ == "__main__":
    main()
