#!/usr/bin/env python3
"""PyRevealed — 1M User Parquet Benchmark.

Generates a large Parquet file, then streams it through the Rust engine
via PyArrow. Demonstrates the full pipeline:

    Parquet on disk  →  PyArrow row-group streaming  →  Rust + Rayon + SCC
    (zstd compressed)   (bounded O(chunk) memory)       (11-core parallel)

Backs up: "We compute all manner of graph-based transitivity violation
scores for about 1 million users, with 20-100 choice occasions each,
in just under 10 minutes."

Usage:
    python benchmarks/bench_parquet_1m.py                # 100K users (quick)
    python benchmarks/bench_parquet_1m.py --users 1000000  # full 1M run
"""

import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def fmt_time(s):
    if s < 1:
        return f"{s*1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s//60)}m {int(s%60)}s"
    return f"{s/3600:.1f}h"


def fmt_bytes(b):
    if b < 1024:
        return f"{b}B"
    if b < 1024**2:
        return f"{b/1024:.1f}KB"
    if b < 1024**3:
        return f"{b/1024**2:.1f}MB"
    return f"{b/1024**3:.2f}GB"


def generate_parquet(path, n_users, t_range, k, seed=42):
    """Generate a wide-format Parquet file with n_users × T_i observations."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    cost_cols = [f"price_{i}" for i in range(k)]
    action_cols = [f"qty_{i}" for i in range(k)]

    writer = None
    schema = None
    total_rows = 0
    batch_users = 10_000  # users per write batch
    user_idx = 0

    while user_idx < n_users:
        batch_end = min(user_idx + batch_users, n_users)
        rows_uid = []
        rows_period = []
        rows_costs = [[] for _ in range(k)]
        rows_actions = [[] for _ in range(k)]

        for uid in range(user_idx, batch_end):
            t = int(rng.integers(t_range[0], t_range[1] + 1))
            for obs in range(t):
                rows_uid.append(f"u{uid:07d}")
                rows_period.append(obs)
                for g in range(k):
                    rows_costs[g].append(float(rng.uniform(0.1, 5.0)))
                    rows_actions[g].append(float(rng.uniform(0.0, 10.0)))
            total_rows += t

        arrays = [pa.array(rows_uid, type=pa.string()), pa.array(rows_period, type=pa.int32())]
        names = ["user_id", "period"]
        for g in range(k):
            arrays.append(pa.array(rows_costs[g], type=pa.float64()))
            names.append(cost_cols[g])
        for g in range(k):
            arrays.append(pa.array(rows_actions[g], type=pa.float64()))
            names.append(action_cols[g])

        table = pa.table(dict(zip(names, arrays)))

        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(str(path), schema, compression="zstd")

        writer.write_table(table)
        user_idx = batch_end

    if writer:
        writer.close()

    return total_rows, cost_cols, action_cols


def main():
    parser = argparse.ArgumentParser(description="PyRevealed Parquet Benchmark")
    parser.add_argument("--users", type=int, default=100_000, help="Number of users")
    parser.add_argument("--k", type=int, default=5, help="Number of goods/categories")
    parser.add_argument("--t-min", type=int, default=20, help="Min observations per user")
    parser.add_argument("--t-max", type=int, default=100, help="Max observations per user")
    parser.add_argument("--chunk", type=int, default=50_000, help="Engine chunk size")
    parser.add_argument("--metrics", type=str, default="garp,ccei,mpi,harp,hm",
                        help="Comma-separated metrics")
    parser.add_argument("--input", type=str, default=None,
                        help="Use existing Parquet file instead of generating")
    parser.add_argument("--output-dir", type=str, default="benchmarks/output",
                        help="Output directory")
    args = parser.parse_args()

    N = args.users
    K = args.k
    T_RANGE = (args.t_min, args.t_max)
    CHUNK = args.chunk
    METRICS = args.metrics.split(",")
    CPU = os.cpu_count() or 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"bench_{N//1000}k_users.parquet"
    results_path = out_dir / f"bench_{N//1000}k_results.parquet"

    cost_cols = [f"price_{i}" for i in range(K)]
    action_cols = [f"qty_{i}" for i in range(K)]

    # ── Header ──────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  PyRevealed — {N:,} User Parquet Benchmark")
    print(f"  PyArrow + Rust + Rayon + Tarjan's SCC")
    print("=" * 72)
    print(f"  Users:       {N:,}  ({T_RANGE[0]}-{T_RANGE[1]} observations each)")
    print(f"  Goods:       {K}")
    print(f"  Metrics:     {', '.join(m.upper() for m in METRICS)}")
    print(f"  Chunk size:  {CHUNK:,} users")
    print(f"  CPU cores:   {CPU}")
    print("=" * 72)

    # ── Phase 1: Generate or load Parquet ──────────────────────────
    if args.input:
        parquet_path = Path(args.input)
        print(f"\n  Phase 1: Loading existing Parquet file...")
        file_size = parquet_path.stat().st_size
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(parquet_path))
        total_rows = pf.metadata.num_rows
        avg_t = total_rows / N if N > 0 else 0
        gen_time = 0.0
        # Auto-detect columns if not specified
        schema_names = [f.name for f in pf.schema_arrow]
        cost_cols = [c for c in schema_names if c.startswith("price_")]
        action_cols = [c for c in schema_names if c.startswith("qty_")]
        K = len(cost_cols)
        print(f"    File:       {parquet_path}")
        print(f"    Rows:       {total_rows:,}")
        print(f"    Size:       {fmt_bytes(file_size)}  (on disk)")
        print(f"    Columns:    {K} price + {K} qty")
    else:
        print(f"\n  Phase 1: Generating Parquet file...")
        t0 = time.perf_counter()
        total_rows, _, _ = generate_parquet(parquet_path, N, T_RANGE, K)
        gen_time = time.perf_counter() - t0
        file_size = parquet_path.stat().st_size
        avg_t = total_rows / N

        print(f"    File:       {parquet_path}")
        print(f"    Rows:       {total_rows:,}  (avg {avg_t:.0f} per user)")
        print(f"    Size:       {fmt_bytes(file_size)}  (zstd compressed)")
        print(f"    Write time: {fmt_time(gen_time)}")

    # ── Phase 2: Stream-analyze via PyArrow ─────────────────────────
    print(f"\n  Phase 2: Streaming analysis (PyArrow → Rust + Rayon)...")
    from pyrevealed.engine import Engine
    from pyrevealed._rust_backend import HAS_PARQUET_RUST

    engine = Engine(metrics=METRICS, chunk_size=CHUNK)
    backend = "Rust Parquet" if HAS_PARQUET_RUST else "PyArrow streaming"
    print(f"    Backend:    {backend}")

    tracemalloc.start()
    t_start = time.perf_counter()

    result_df = engine.analyze_parquet(
        parquet_path,
        user_col="user_id",
        cost_cols=cost_cols,
        action_cols=action_cols,
    )

    total_time = time.perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_users_out = len(result_df)
    throughput = n_users_out / total_time

    print(f"    Users out:  {n_users_out:,}")
    print(f"    Total time: {fmt_time(total_time)}")
    print(f"    Throughput: {throughput:,.0f} users/sec")
    print(f"    Peak mem:   {fmt_bytes(peak_mem)}")

    # ── Phase 3: Results summary ────────────────────────────────────
    print(f"\n  Phase 3: Score distributions")
    print("  " + "-" * 70)

    n_consistent = result_df["is_garp"].sum()
    pct = n_consistent / n_users_out * 100
    print(f"    GARP-consistent:  {n_consistent:,} / {n_users_out:,}  ({pct:.1f}%)")

    if "ccei" in result_df.columns:
        ccei = result_df["ccei"]
        print(f"    CCEI:   mean={ccei.mean():.4f}  median={ccei.median():.4f}  "
              f"std={ccei.std():.4f}  P10={ccei.quantile(0.1):.3f}  P90={ccei.quantile(0.9):.3f}")

    if "mpi" in result_df.columns:
        mpi = result_df["mpi"]
        mpi_nonzero = mpi[mpi > 0]
        print(f"    MPI:    mean={mpi.mean():.4f}  "
              f"nonzero={len(mpi_nonzero):,} ({len(mpi_nonzero)/n_users_out*100:.1f}%)  "
              f"max={mpi.max():.4f}")

    if "is_harp" in result_df.columns:
        n_harp = result_df["is_harp"].sum()
        print(f"    HARP:   {n_harp:,} consistent ({n_harp/n_users_out*100:.1f}%)")

    if "hm_total" in result_df.columns:
        hm_frac = result_df["hm_consistent"] / result_df["hm_total"].clip(lower=1)
        print(f"    HM:     mean fraction={hm_frac.mean():.4f}  "
              f"fully consistent={int((hm_frac == 1.0).sum()):,}")

    if "compute_time_us" in result_df.columns:
        ct = result_df["compute_time_us"]
        print(f"    Per-user compute:  mean={ct.mean():.0f}μs  "
              f"median={ct.median():.0f}μs  P99={ct.quantile(0.99):.0f}μs")

    # ── Phase 4: Write results Parquet ──────────────────────────────
    print(f"\n  Phase 4: Writing results to Parquet...")
    result_df.to_parquet(results_path, compression="zstd")
    result_size = results_path.stat().st_size
    print(f"    File:       {results_path}")
    print(f"    Size:       {fmt_bytes(result_size)}")

    # ── Summary card ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"  Input:       {fmt_bytes(file_size)} Parquet  →  {total_rows:,} rows  →  {n_users_out:,} users")
    print(f"  Output:      {fmt_bytes(result_size)} Parquet  ({len(result_df.columns)} score columns)")
    print(f"  Metrics:     {', '.join(m.upper() for m in METRICS)}")
    print(f"  Pipeline:    PyArrow row-groups → Rust + Rayon + Tarjan SCC")
    print(f"  Wall time:   {fmt_time(total_time)}  ({throughput:,.0f} users/sec)")
    print(f"  Peak memory: {fmt_bytes(peak_mem)}")
    print()
    print(f"  Scaling projections at {throughput:,.0f} users/sec:")
    for label, count in [("1M users", 1_000_000), ("10M users", 10_000_000)]:
        t = count / throughput
        print(f"    {label:>12s}:  {fmt_time(t)}")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
