from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable

import matplotlib


# Use non-interactive backend for headless environments (RTD/CI)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# Ensure local source is importable without installation
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


@dataclass
class Config:
    """Configuration for performance plot generation.

    Keep values modest for "light" mode so it can run on CI/RTD
    fallback Python backend without the Rust extension.
    """

    # Plot 1 (throughput)
    user_counts: list[int]

    # Plot 2 (per-user time)
    t_values: list[int]
    n_users_per_t: int

    # Plot 3 (memory)
    memory_user_counts: list[int]
    chunk_size: int

    # Data generation
    k_goods: int = 5
    seed: int = 42

    # Output directory (defaults to docs/_static)
    out_dir: str = os.path.join(REPO_ROOT, "docs", "_static")


def light_config() -> Config:
    return Config(
        user_counts=[1_000, 5_000, 10_000],
        t_values=[20, 50, 100],
        n_users_per_t=500,
        memory_user_counts=[10_000, 50_000],
        chunk_size=10_000,
    )


def full_config() -> Config:
    return Config(
        user_counts=[1_000, 5_000, 10_000, 50_000, 100_000],
        t_values=[20, 50, 100, 200, 500],
        n_users_per_t=2_000,
        memory_user_counts=[10_000, 50_000, 100_000, 500_000],
        chunk_size=50_000,
    )


# PyData color palette (matches RTD theme)
COLORS = {
    "garp": "#0173B2",
    "ccei": "#DE8F05",
    "mpi": "#029E73",
    "harp": "#D55E00",
    "all": "#CC78BC",
    "memory": "#0173B2",
}


def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def gen_users(n: int, t_range: tuple[int, int] = (20, 100), k: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    users = []
    for _ in range(n):
        t = int(rng.integers(t_range[0], t_range[1] + 1))
        p = np.ascontiguousarray(rng.random((t, k)) + 0.1, dtype=np.float64)
        q = np.ascontiguousarray(rng.random((t, k)) + 0.1, dtype=np.float64)
        users.append((p, q))
    return users


def plot_style(ax, title: str, xlabel: str, ylabel: str, legend: bool = True) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if legend:
        ax.legend(fontsize=10, frameon=True, fancybox=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def logspace_ticks(values: Iterable[int]) -> None:
    import matplotlib.ticker as mticker

    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

