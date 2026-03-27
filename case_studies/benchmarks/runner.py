#!/usr/bin/env python3
"""ML Benchmark Runner: Revealed preference features as predictive signals.

Runs three-way comparison (RP-only vs Baseline-only vs Combined) across
multiple real-world datasets and prediction targets.

Usage:
    python case_studies/benchmarks/runner.py                     # All available datasets
    python case_studies/benchmarks/runner.py --datasets dunnhumby,uci_retail
    python case_studies/benchmarks/runner.py --datasets dunnhumby --max-users 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from case_studies.benchmarks.core.evaluation import BenchmarkResult
from case_studies.benchmarks.core.reporting import print_summary, save_results, generate_plots


AVAILABLE_DATASETS = {
    # Budget-based (real prices)
    "dunnhumby": "case_studies.benchmarks.datasets.dunnhumby_bench",
    "open_ecommerce": "case_studies.benchmarks.datasets.open_ecommerce_bench",
    "hm": "case_studies.benchmarks.datasets.hm_bench",
    # Budget-based (uniform prices)
    "instacart": "case_studies.benchmarks.datasets.instacart_bench",
    # Menu-based
    "rees46": "case_studies.benchmarks.datasets.rees46_bench",
    "taobao": "case_studies.benchmarks.datasets.taobao_bench",
}


def run_dataset(name: str, max_users: int | None = None) -> list[BenchmarkResult]:
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
    elif name in ("instacart", "rees46", "hm", "taobao"):
        kwargs["max_users"] = max_users or 50000

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
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: case_studies/benchmarks/output/)",
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

    print("=" * 70)
    print(" ML BENCHMARK: Revealed Preference Features as Predictive Signals")
    print("=" * 70)
    print(f"\n  Datasets: {', '.join(dataset_names)}")
    print(f"  Max users: {args.max_users or 'unlimited'}")
    print(f"  Output: {output_dir}")

    all_results: list[BenchmarkResult] = []
    start = time.time()

    for name in dataset_names:
        t0 = time.time()
        results = run_dataset(name, args.max_users)
        elapsed = time.time() - t0
        print(f"  [{name}] Completed in {elapsed:.1f}s ({len(results)} targets)")
        all_results.extend(results)

    total_time = time.time() - start
    print(f"\nTotal time: {total_time:.1f}s")

    if all_results:
        print_summary(all_results)
        save_results(all_results, output_dir)
        generate_plots(all_results, output_dir)
    else:
        print("\nNo results — check that at least one dataset is available.")


if __name__ == "__main__":
    main()
