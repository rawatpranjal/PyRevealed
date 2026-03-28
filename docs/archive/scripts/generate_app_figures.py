#!/usr/bin/env python3
"""Generate 2x2 figure panels for the three application documentation pages."""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

plt.switch_backend("Agg")

OUTPUT_DIR = Path(__file__).parent.parent / "images"

# Shared style
STYLE = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

COLORS = {
    "rational": "#2ecc71",
    "noisy": "#f39c12",
    "irrational": "#e74c3c",
    "green": "#2ecc71",
    "orange": "#f39c12",
    "red": "#e74c3c",
    "blue": "#3498db",
    "purple": "#9b59b6",
    "gray": "#95a5a6",
}


def _apply_style():
    plt.rcParams.update(STYLE)


# NOTE: Content identical to docs/generate_app_figures.py at time of archive.
# Paths adjusted to save under docs/images from archive/scripts/ location.

# ---------------------------------------------------------------------------
# Panel 1: Grocery Scanner
# ---------------------------------------------------------------------------


def generate_grocery_panel():
    """Generate 2x2 panel for the grocery scanner application."""
    from prefgraph.datasets import load_demo
    from prefgraph.engine import Engine
    from prefgraph import BehaviorLog, compute_integrity_score, recover_utility

    _apply_style()

    users = load_demo(n_users=100, n_obs=30, n_goods=5, seed=42)
    engine = Engine(metrics=["garp", "ccei", "mpi", "hm"])
    results = engine.analyze_arrays(users)

    ccei_scores = [r.ccei for r in results]
    mpi_scores = [r.mpi for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) CCEI Distribution
    ax = axes[0, 0]
    bins = np.linspace(0.3, 1.0, 30)
    n_vals, bin_edges, patches = ax.hist(ccei_scores, bins=bins, edgecolor="white", linewidth=0.5)
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge >= 0.95:
            patch.set_facecolor(COLORS["green"])
        elif left_edge >= 0.80:
            patch.set_facecolor(COLORS["orange"])
        else:
            patch.set_facecolor(COLORS["red"])
    ax.axvline(0.95, color=COLORS["green"], linestyle="--", alpha=0.7, label="Consistent (>0.95)")
    ax.axvline(0.80, color=COLORS["red"], linestyle="--", alpha=0.7, label="Erratic (<0.80)")
    ax.axvline(np.mean(ccei_scores), color="black", linestyle=":", alpha=0.5, label=f"Mean={np.mean(ccei_scores):.2f}")
    ax.set_xlabel("CCEI Score")
    ax.set_ylabel("Number of Users")
    ax.set_title("(a) CCEI Score Distribution")
    ax.legend(fontsize=8)

    # (b) CCEI vs MPI Scatter
    ax = axes[0, 1]
    for i, (c, m) in enumerate(zip(ccei_scores, mpi_scores)):
        if i < 40:
            color, label = COLORS["rational"], "Rational"
        elif i < 80:
            color, label = COLORS["noisy"], "Noisy"
        else:
            color, label = COLORS["irrational"], "Irrational"
        ax.scatter(c, m, c=color, alpha=0.6, s=30, edgecolors="white", linewidth=0.3)
    for label, color in [("Rational", COLORS["rational"]), ("Noisy", COLORS["noisy"]), ("Irrational", COLORS["irrational"])]:
        ax.scatter([], [], c=color, label=label, s=30)
    ax.set_xlabel("CCEI (Efficiency)")
    ax.set_ylabel("MPI (Exploitability)")
    ax.set_title("(b) Efficiency vs Exploitability")
    ax.legend(fontsize=8)

    # (c) Rolling-Window CCEI trajectories (representative users)
    ax = axes[1, 0]
    from prefgraph import BehaviorLog as _BehaviorLog
    window = 15
    representative = [0, 50, 85]
    labels_map = {0: "Rational", 50: "Noisy", 85: "Irrational"}
    colors_map = {0: COLORS["rational"], 50: COLORS["noisy"], 85: COLORS["irrational"]}
    for uid in representative:
        prices, quantities = users[uid]
        trajectory = []
        for start in range(0, len(prices) - window + 1):
            wlog = _BehaviorLog(cost_vectors=prices[start : start + window], action_vectors=quantities[start : start + window])
            try:
                from prefgraph import compute_integrity_score as _cis
                score = _cis(wlog).efficiency_index
            except Exception:
                score = np.nan
            trajectory.append(score)
        ax.plot(range(len(trajectory)), trajectory, marker="o", markersize=4, label=labels_map[uid], color=colors_map[uid], linewidth=2)
    ax.set_xlabel("Window Start Index")
    ax.set_ylabel("CCEI")
    ax.set_title(f"(c) Rolling-Window CCEI (window={window})")
    ax.legend(fontsize=8)
    ax.set_ylim(0.3, 1.05)

    # (d) Recovered Utility (Afriat LP)
    ax = axes[1, 1]
    prices_0, quantities_0 = users[0]
    log_0 = _BehaviorLog(cost_vectors=prices_0, action_vectors=quantities_0)
    try:
        from prefgraph import recover_utility as _ru
        u_result = _ru(log_0)
        u_vals = u_result.utility_values
        obs_idx = np.arange(len(u_vals))
        ax.bar(obs_idx, u_vals, color=COLORS["blue"], edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Observation (Shopping Week)")
        ax.set_ylabel("Recovered Utility")
        ax.set_title("(d) Afriat LP Utility Recovery")
    except Exception as e:
        ax.text(0.5, 0.5, f"Utility recovery\\nnot available\\n({e})", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("(d) Afriat LP Utility Recovery")

    fig.suptitle("Grocery Scanner: Revealed Preference Analysis (N=100 households)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(pad=2.0)
    out = OUTPUT_DIR / "app_grocery_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# (Other panel generators preserved; omitted here for brevity)

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    generate_grocery_panel()


if __name__ == "__main__":
    main()
