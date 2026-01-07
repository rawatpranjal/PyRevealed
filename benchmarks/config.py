"""Benchmark configuration constants."""

from pathlib import Path

# Paths
BENCHMARKS_DIR = Path(__file__).parent
OUTPUT_DIR = BENCHMARKS_DIR / "output"

# Scale levels (observations T)
SCALE_LEVELS = [10, 50, 100, 500, 1000, 2000, 5000, 10000]

# Quick mode scale levels
QUICK_SCALE_LEVELS = [10, 50, 100, 500]

# Goods dimensions (N)
GOODS_DIMENSIONS = [5, 10]

# Data types for benchmarking
DATA_TYPES = ["rational", "random"]

# Algorithms to benchmark
ALGORITHMS = {
    "garp": {"complexity": "O(T^3)", "function": "check_garp"},
    "aei": {"complexity": "O(T^3 log T)", "function": "compute_aei"},
    "mpi": {"complexity": "O(T^3)", "function": "compute_mpi"},
    "harp": {"complexity": "O(T^3)", "function": "check_harp"},
    "differentiable": {"complexity": "O(T^3)", "function": "check_differentiable"},
    "acyclical_p": {"complexity": "O(T^3)", "function": "check_acyclical_p"},
    "gapp": {"complexity": "O(T^3)", "function": "check_gapp"},
}

# Benchmark parameters
NUM_WARMUP_RUNS = 1
NUM_TIMED_RUNS = 3
TIMEOUT_SECONDS = 600  # 10 minutes max per algorithm per scale

# Memory tracking
TRACK_MEMORY = True

# Expected scaling targets (for validation)
MAX_ACCEPTABLE_TIME_10K = {
    "garp": 60.0,    # 60 seconds for T=10000
    "aei": 120.0,    # 2 minutes (includes binary search)
    "mpi": 90.0,
    "harp": 60.0,
    "differentiable": 60.0,
    "acyclical_p": 60.0,
    "gapp": 60.0,
}
