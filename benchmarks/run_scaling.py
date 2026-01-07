#!/usr/bin/env python3
"""
Scaling Benchmark Suite for PyRevealed Algorithms

Tests algorithm performance scaling up to 10K observations.

Usage:
    python benchmarks/run_scaling.py           # Full benchmark
    python benchmarks/run_scaling.py --quick   # Quick test (500 obs max)
    python benchmarks/run_scaling.py --algorithm garp  # Single algorithm
"""

import argparse
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.config import (
    ALGORITHMS,
    GOODS_DIMENSIONS,
    NUM_TIMED_RUNS,
    NUM_WARMUP_RUNS,
    OUTPUT_DIR,
    QUICK_SCALE_LEVELS,
    SCALE_LEVELS,
    TRACK_MEMORY,
)
from benchmarks.generators import generate_benchmark_data

from pyrevealed import ConsumerSession
from pyrevealed.algorithms.garp import check_garp
from pyrevealed.algorithms.aei import compute_aei
from pyrevealed.algorithms.mpi import compute_mpi
from pyrevealed.algorithms.harp import check_harp
from pyrevealed.algorithms.differentiable import check_differentiable
from pyrevealed.algorithms.acyclical_p import check_acyclical_p
from pyrevealed.algorithms.gapp import check_gapp


@dataclass
class BenchmarkResult:
    """Result for a single benchmark run."""

    algorithm: str
    n_observations: int
    n_goods: int
    data_type: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    peak_memory_mb: float
    throughput_obs_per_sec: float
    success: bool
    error_message: str = ""


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    results: list[BenchmarkResult] = field(default_factory=list)
    total_time_seconds: float = 0.0

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def to_csv(self, path: Path) -> None:
        """Export results to CSV."""
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(
                [
                    "algorithm",
                    "n_observations",
                    "n_goods",
                    "data_type",
                    "mean_time_ms",
                    "std_time_ms",
                    "min_time_ms",
                    "max_time_ms",
                    "peak_memory_mb",
                    "throughput_obs_per_sec",
                    "success",
                    "error_message",
                ]
            )
            # Data
            for r in self.results:
                writer.writerow(
                    [
                        r.algorithm,
                        r.n_observations,
                        r.n_goods,
                        r.data_type,
                        f"{r.mean_time_ms:.2f}",
                        f"{r.std_time_ms:.2f}",
                        f"{r.min_time_ms:.2f}",
                        f"{r.max_time_ms:.2f}",
                        f"{r.peak_memory_mb:.2f}",
                        f"{r.throughput_obs_per_sec:.2f}",
                        r.success,
                        r.error_message,
                    ]
                )


def get_algorithm_func(algorithm_name: str) -> Callable:
    """Get algorithm function by name."""
    funcs = {
        "garp": lambda s: check_garp(s),
        "aei": lambda s: compute_aei(s, tolerance=1e-3, max_iterations=20),
        "mpi": lambda s: compute_mpi(s),
        "harp": lambda s: check_harp(s),
        "differentiable": lambda s: check_differentiable(s),
        "acyclical_p": lambda s: check_acyclical_p(s),
        "gapp": lambda s: check_gapp(s),
    }
    return funcs[algorithm_name]


def run_single_benchmark(
    algorithm_name: str,
    n_observations: int,
    n_goods: int,
    data_type: str = "rational",
    seed: int = 42,
) -> BenchmarkResult:
    """Run benchmark for a single configuration."""

    # Generate data
    prices, quantities = generate_benchmark_data(
        n_observations, n_goods, data_type, seed
    )
    session = ConsumerSession(prices=prices, quantities=quantities)

    # Get algorithm function
    algo_func = get_algorithm_func(algorithm_name)

    times_ms: list[float] = []
    peak_memory_mb = 0.0

    try:
        # Warmup runs
        for _ in range(NUM_WARMUP_RUNS):
            algo_func(session)

        # Timed runs
        for _ in range(NUM_TIMED_RUNS):
            if TRACK_MEMORY:
                tracemalloc.start()

            start = time.perf_counter()
            algo_func(session)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if TRACK_MEMORY:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_memory_mb = max(peak_memory_mb, peak / 1024 / 1024)

            times_ms.append(elapsed_ms)

        times = np.array(times_ms)
        throughput = (n_observations * 1000) / np.mean(times)  # obs/sec

        return BenchmarkResult(
            algorithm=algorithm_name,
            n_observations=n_observations,
            n_goods=n_goods,
            data_type=data_type,
            mean_time_ms=float(np.mean(times)),
            std_time_ms=float(np.std(times)),
            min_time_ms=float(np.min(times)),
            max_time_ms=float(np.max(times)),
            peak_memory_mb=peak_memory_mb,
            throughput_obs_per_sec=throughput,
            success=True,
        )

    except Exception as e:
        return BenchmarkResult(
            algorithm=algorithm_name,
            n_observations=n_observations,
            n_goods=n_goods,
            data_type=data_type,
            mean_time_ms=-1,
            std_time_ms=-1,
            min_time_ms=-1,
            max_time_ms=-1,
            peak_memory_mb=-1,
            throughput_obs_per_sec=-1,
            success=False,
            error_message=str(e),
        )


def run_full_benchmark(
    algorithms: list[str] | None = None,
    scale_levels: list[int] | None = None,
    goods_dims: list[int] | None = None,
    data_types: list[str] | None = None,
    quick_mode: bool = False,
) -> BenchmarkSuite:
    """Run the full benchmark suite."""

    # Use defaults or quick mode settings
    algorithms = algorithms or list(ALGORITHMS.keys())
    scale_levels = scale_levels or (QUICK_SCALE_LEVELS if quick_mode else SCALE_LEVELS)
    goods_dims = goods_dims or [10]  # Default N=10
    data_types = data_types or ["rational"]  # Default to rational for timing

    suite = BenchmarkSuite()
    total_configs = len(algorithms) * len(scale_levels) * len(goods_dims) * len(data_types)

    print_banner("PYREVEALED SCALING BENCHMARK SUITE")
    print(f"  Algorithms: {algorithms}")
    print(f"  Scale levels (T): {scale_levels}")
    print(f"  Goods dimensions (N): {goods_dims}")
    print(f"  Data types: {data_types}")
    print(f"  Total configurations: {total_configs}")

    overall_start = time.perf_counter()
    config_idx = 0

    for algo in algorithms:
        print_banner(f"Benchmarking: {algo.upper()}", char="-")

        for T in scale_levels:
            for N in goods_dims:
                for dtype in data_types:
                    config_idx += 1
                    print(
                        f"  [{config_idx}/{total_configs}] T={T}, N={N}, type={dtype}...",
                        end=" ",
                        flush=True,
                    )

                    result = run_single_benchmark(algo, T, N, dtype)
                    suite.add(result)

                    if result.success:
                        print(f"{result.mean_time_ms:.1f}ms (+-{result.std_time_ms:.1f})")
                    else:
                        print(f"FAILED: {result.error_message}")

    suite.total_time_seconds = time.perf_counter() - overall_start

    print_banner("BENCHMARK COMPLETE")
    print(f"  Total time: {suite.total_time_seconds:.1f}s")
    print(f"  Results: {len(suite.results)}")

    return suite


def print_banner(text: str, char: str = "=", width: int = 70) -> None:
    """Print a banner line."""
    print()
    print(char * width)
    print(f" {text}")
    print(char * width)


def print_summary_table(suite: BenchmarkSuite) -> None:
    """Print a summary table of results."""
    print_banner("SUMMARY TABLE")

    # Group by algorithm
    algos = sorted(set(r.algorithm for r in suite.results))
    scales = sorted(set(r.n_observations for r in suite.results))

    # Print header
    header = "Algorithm".ljust(15)
    for T in scales:
        header += f"T={T}".rjust(12)
    print(header)
    print("-" * len(header))

    # Print rows
    for algo in algos:
        row = algo.ljust(15)
        for T in scales:
            matching = [r for r in suite.results if r.algorithm == algo and r.n_observations == T]
            if matching and matching[0].success:
                time_ms = matching[0].mean_time_ms
                if time_ms >= 1000:
                    row += f"{time_ms/1000:.1f}s".rjust(12)
                else:
                    row += f"{time_ms:.0f}ms".rjust(12)
            else:
                row += "FAIL".rjust(12)
        print(row)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PyRevealed Scaling Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick mode (small scale)")
    parser.add_argument("--algorithm", type=str, help="Single algorithm to benchmark")
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_DIR / "scaling_results.csv")
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    algorithms = [args.algorithm] if args.algorithm else None

    suite = run_full_benchmark(
        algorithms=algorithms,
        quick_mode=args.quick,
    )

    # Print summary
    print_summary_table(suite)

    # Save results
    output_path = Path(args.output)
    suite.to_csv(output_path)
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
