#!/usr/bin/env python3
"""Generate performance plots for RTD under a configurable profile.

Exposes three functions to build the figures used on docs/performance.rst:
  - throughput vs number of users
  - per-user time vs observations T for key metrics
  - memory vs users under chunked streaming

Use docs/perf/cli.py for a simple entry point.
"""

from __future__ import annotations

import os
import time
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    COLORS,
    Config,
    ensure_out_dir,
    gen_users,
    plot_style,
)


def _engine(metrics: Sequence[str], chunk_size: int):
    # Local import to honor sys.path bootstrapping in utils
    from prefgraph.engine import Engine

    return Engine(metrics=list(metrics), chunk_size=chunk_size)


def generate_throughput_plot(cfg: Config) -> str:
    """Plot 1: Throughput vs number of users."""
    out_dir = ensure_out_dir(cfg.out_dir)
    user_counts = cfg.user_counts

    configs = [
        (["garp"], "GARP only", COLORS["garp"]),
        (["garp", "ccei"], "GARP + CCEI", COLORS["ccei"]),
        (["garp", "ccei", "mpi", "harp"], "All metrics", COLORS["all"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for metrics, label, color in configs:
        engine = _engine(metrics, cfg.chunk_size)
        throughputs: list[float] = []

        for i, n in enumerate(user_counts):
            users = gen_users(n, k=cfg.k_goods, seed=cfg.seed)
            # Warmup on smallest n to trigger imports/compilation paths
            if i == 0:
                engine.analyze_arrays(gen_users(50, k=cfg.k_goods, seed=cfg.seed))

            t0 = time.perf_counter()
            engine.analyze_arrays(users)
            elapsed = time.perf_counter() - t0
            throughputs.append(n / max(elapsed, 1e-9))

        ax.plot(user_counts, throughputs, "o-", color=color, label=label,
                linewidth=2, markersize=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    plot_style(ax, "Throughput scales with users", "Number of users", "Users / second")

    fig.tight_layout()
    path = os.path.join(out_dir, "perf_throughput.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_per_user_plot(cfg: Config) -> str:
    """Plot 2: Per-user time vs observations T."""
    out_dir = ensure_out_dir(cfg.out_dir)
    t_values = cfg.t_values
    n_users = cfg.n_users_per_t

    metric_configs = [
        (["garp"], "GARP", COLORS["garp"]),
        (["garp", "ccei"], "CCEI", COLORS["ccei"]),
        (["garp", "mpi"], "MPI", COLORS["mpi"]),
        (["garp", "harp"], "HARP", COLORS["harp"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for metrics, label, color in metric_configs:
        engine = _engine(metrics, cfg.chunk_size)
        us_per_user = []

        for t_val in t_values:
            users = gen_users(n_users, t_range=(t_val, t_val), k=cfg.k_goods, seed=cfg.seed)
            engine.analyze_arrays(gen_users(20, t_range=(t_val, t_val), k=cfg.k_goods, seed=cfg.seed))

            t0 = time.perf_counter()
            engine.analyze_arrays(users)
            elapsed = time.perf_counter() - t0
            us = elapsed / n_users * 1e6
            us_per_user.append(us)

        ax.plot(t_values, us_per_user, "o-", color=color, label=label,
                linewidth=2, markersize=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    plot_style(ax, "Per-user compute time by metric", "Observations per user (T)", "Microseconds per user")

    fig.tight_layout()
    path = os.path.join(out_dir, "perf_per_user.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_memory_plot(cfg: Config) -> str:
    """Plot 3: Peak memory vs users under streaming."""
    import tracemalloc

    out_dir = ensure_out_dir(cfg.out_dir)
    user_counts = cfg.memory_user_counts

    engine = _engine(["garp", "ccei", "mpi", "harp"], cfg.chunk_size)
    peak_mems = []

    for n in user_counts:
        tracemalloc.start()

        # Stream in chunks to bound memory
        rng = np.random.default_rng(cfg.seed)
        processed = 0
        while processed < n:
            batch = int(min(cfg.chunk_size, n - processed))
            chunk = []
            for _ in range(batch):
                t = int(rng.integers(cfg.t_values[0], cfg.t_values[-1] + 1))
                p = np.ascontiguousarray(rng.random((t, cfg.k_goods)) + 0.1, dtype=np.float64)
                q = np.ascontiguousarray(rng.random((t, cfg.k_goods)) + 0.1, dtype=np.float64)
                chunk.append((p, q))
            engine.analyze_arrays(chunk)
            processed += batch
            del chunk

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mems.append(peak / 1e6)  # MB

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(user_counts, peak_mems, "o-", color=COLORS["memory"], linewidth=2, markersize=8,
            label=f"With streaming (chunks of {cfg.chunk_size:,})")

    # Reference line: approximate naive memory (no streaming, doubles arrays)
    naive_mems = [n * 60 * cfg.k_goods * 8 * 2 / 1e6 for n in user_counts]
    ax.plot(user_counts, naive_mems, "--", color="#999999", linewidth=1,
            label="Without streaming (all users in RAM)", alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    plot_style(ax, "Memory stays bounded under streaming", "Number of users", "Peak memory (MB)")

    fig.tight_layout()
    path = os.path.join(out_dir, "perf_memory.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all(cfg: Config) -> list[str]:
    paths = [
        generate_throughput_plot(cfg),
        generate_per_user_plot(cfg),
        generate_memory_plot(cfg),
    ]
    return paths

