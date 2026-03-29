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
}


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
    elif name in ("instacart_v2_menu", "rees46", "hm", "taobao", "retailrocket", "tenrec", "yoochoose", "mind"):
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
        help=f"Comma-separated dataset names or 'all'. Available: {', '.join(AVAILABLE_DATASETS)}",
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
    args = parser.parse_args()

    if args.datasets == "all":
        dataset_names = list(AVAILABLE_DATASETS.keys())
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

    print("=" * 70)
    print(" ML BENCHMARK: Revealed Preference Features as Predictive Signals")
    print("=" * 70)
    print(f"\n  Datasets: {', '.join(dataset_names)}")
    print(f"  Max users: {args.max_users or 'unlimited'}")
    print(f"  Output: {output_dir}")
    if args.skip_existing:
        print(f"  Mode: skip-existing (use cached results where available)")

    all_results: list[BenchmarkResult] = []
    start = time.time()

    for name in dataset_names:
        display_name = DATASET_DISPLAY_NAMES.get(name, name)

        # Check cache if --skip-existing
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

        # Persist per-dataset results immediately
        if results:
            save_dataset_results(results, results[0].dataset, output_dir)

        all_results.extend(results)

    total_time = time.time() - start
    print(f"\nTotal time: {total_time:.1f}s")

    if all_results:
        print_summary(all_results)
        save_results(all_results, output_dir)
        generate_plots(all_results, output_dir)
    else:
        print("\nNo results - check that at least one dataset is available.")


if __name__ == "__main__":
    main()
