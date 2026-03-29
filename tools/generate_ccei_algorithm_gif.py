"""
CCEI (Afriat Efficiency Index) algorithm GIF — visual-first, no text flicker.

Shows two budget lines with bundles that create a GARP violation. Budget
lines progressively shrink (e decreases from 1.0) until the violation
disappears. CCEI value updates on the side.

Uses PrefGraph's actual compute_integrity_score to get the real CCEI.

Usage:
    python3 tools/generate_ccei_algorithm_gif.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation
from pathlib import Path
from prefgraph import BehaviorLog
from prefgraph.algorithms.aei import compute_integrity_score

OUT_DIR = Path("docs/_static")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 150

# PrefGraph style spec
COL_BG = "#fafafa"
COL_BLUE = "#2563eb"
COL_PURPLE = "#8e44ad"
COL_RED = "#e74c3c"
COL_GREEN = "#27ae60"
COL_TEXT = "#333333"
COL_SUBTEXT = "#666666"

# Two shopping trips with a GARP violation
# Trip 1: prices (1, 2), quantities (2, 3) → expenditure = 8
# Trip 2: prices (2, 1), quantities (3, 2) → expenditure = 8
# Each bundle was affordable at the other's prices → contradiction
PRICES = np.array([[1.0, 2.0], [2.0, 1.0]])
QUANTS = np.array([[2.0, 3.0], [3.0, 2.0]])
EXPENDITURES = np.sum(PRICES * QUANTS, axis=1)  # [8, 8]
TRIP_COLORS = [COL_BLUE, COL_PURPLE]

# Compute real CCEI
log = BehaviorLog(prices=PRICES, quantities=QUANTS)
ccei_result = compute_integrity_score(log)
CCEI_FINAL = ccei_result.score()

# Animation: 40 frames
# Phase 1: show full budgets + bundles + violation arrows (10 frames)
# Phase 2: shrink budgets smoothly from e=1.0 to e=CCEI (20 frames)
# Phase 3: hold at CCEI (10 frames)
FRAMES_SHOW = 10
FRAMES_SHRINK = 20
FRAMES_HOLD = 10
TOTAL_FRAMES = FRAMES_SHOW + FRAMES_SHRINK + FRAMES_HOLD


def draw_budget_line(ax, trip_idx, e=1.0, ghost=False):
    """Draw budget line at efficiency level e."""
    p0, p1 = PRICES[trip_idx]
    budget = EXPENDITURES[trip_idx] * e
    x_int = budget / p0
    y_int = budget / p1
    alpha = 0.12 if ghost else 0.5
    lw = 1.0 if ghost else 2.5
    ls = "--" if ghost else "-"
    color = TRIP_COLORS[trip_idx]
    ax.plot([0, x_int], [y_int, 0], color=color, lw=lw, alpha=alpha, ls=ls)
    if not ghost:
        ax.fill_between([0, x_int], [y_int, 0], alpha=0.06, color=color)


def draw_arrow(ax, x0, y0, x1, y1, color, lw=1.5):
    dx, dy = x1 - x0, y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    if length < 0.01:
        return
    shrink = 0.35
    sx = x0 + shrink * dx / length
    sy = y0 + shrink * dy / length
    ex = x1 - shrink * dx / length
    ey = y1 - shrink * dy / length
    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0))


def generate_gif():
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=COL_BG)
    fig.subplots_adjust(left=0.10, right=0.70, bottom=0.10, top=0.95)

    def update(frame):
        ax.clear()
        ax.set_facecolor(COL_BG)
        ax.set_xlim(-0.3, 9.0)
        ax.set_ylim(-0.3, 5.5)
        ax.set_xlabel("Good 1", fontsize=10, color=COL_SUBTEXT)
        ax.set_ylabel("Good 2", fontsize=10, color=COL_SUBTEXT)
        ax.tick_params(labelsize=8, colors=COL_SUBTEXT)
        ax.grid(True, alpha=0.1, color="#e0e0e0")
        for spine in ax.spines.values():
            spine.set_color("#e0e0e0")

        # Compute current e
        if frame < FRAMES_SHOW:
            e = 1.0
            show_arrows = True
        elif frame < FRAMES_SHOW + FRAMES_SHRINK:
            progress = (frame - FRAMES_SHOW) / FRAMES_SHRINK
            e = 1.0 - progress * (1.0 - CCEI_FINAL)
            show_arrows = False
        else:
            e = CCEI_FINAL
            show_arrows = False

        # Ghost lines at e=1.0 when shrinking
        if e < 1.0:
            draw_budget_line(ax, 0, e=1.0, ghost=True)
            draw_budget_line(ax, 1, e=1.0, ghost=True)

        # Current budget lines
        draw_budget_line(ax, 0, e=e)
        draw_budget_line(ax, 1, e=e)

        # Bundles (always shown)
        for t in range(2):
            ax.scatter(QUANTS[t, 0], QUANTS[t, 1], color=TRIP_COLORS[t],
                       s=120, zorder=5, edgecolors="white", lw=2)

        # Preference arrows (only in show phase)
        if show_arrows:
            arrow_color = COL_RED
            draw_arrow(ax, QUANTS[0, 0], QUANTS[0, 1],
                       QUANTS[1, 0], QUANTS[1, 1], color=arrow_color, lw=2.0)
            draw_arrow(ax, QUANTS[1, 0], QUANTS[1, 1],
                       QUANTS[0, 0], QUANTS[0, 1], color=arrow_color, lw=2.0)

        # CCEI display on the right side (in figure coords)
        x_r = 0.82
        is_final = frame >= FRAMES_SHOW + FRAMES_SHRINK

        fig.texts.clear()
        fig.text(x_r, 0.75, "CCEI", ha="center", fontsize=11,
                 fontweight="bold", color=COL_TEXT)
        ccei_color = COL_GREEN if is_final else COL_BLUE
        fig.text(x_r, 0.60, f"{e:.3f}", ha="center", fontsize=28,
                 fontweight="bold", color=ccei_color, fontfamily="monospace")

        if frame < FRAMES_SHOW:
            fig.text(x_r, 0.45, "Violation\ndetected", ha="center",
                     fontsize=9, color=COL_RED)
        elif not is_final:
            fig.text(x_r, 0.45, "Shrinking\nbudgets...", ha="center",
                     fontsize=9, color=COL_SUBTEXT)
        else:
            fig.text(x_r, 0.45, "Violation\nresolved", ha="center",
                     fontsize=9, fontweight="bold", color=COL_GREEN)

    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=250)
    out_path = OUT_DIR / "ccei_algorithm.gif"
    anim.save(str(out_path), writer="pillow", dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB, "
          f"{TOTAL_FRAMES} frames, CCEI={CCEI_FINAL:.3f})")


if __name__ == "__main__":
    generate_gif()
