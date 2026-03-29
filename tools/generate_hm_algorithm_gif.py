"""
Houtman-Maks algorithm GIF — visual-first, no text flicker.

Shows a preference graph with violations. Observations are removed one by
one (greedy FVS) to break cycles. Removed nodes fade out, cycle edges turn
from red to blue as violations resolve. HM fraction updates on the side.

Usage:
    python3 tools/generate_hm_algorithm_gif.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.animation import FuncAnimation
from pathlib import Path

OUT_DIR = Path("docs/_static")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 150

# PrefGraph style spec
COL_BG = "#fafafa"
COL_NODE = "#2563eb"
COL_EDGE = "#3b82f6"
COL_RED = "#e74c3c"
COL_FADED = "#d0d0d0"
COL_TEXT = "#333333"
COL_SUBTEXT = "#666666"

# 5 nodes with edges that form 2 cycles
N = 5
LABELS = [f"$x_{i+1}$" for i in range(N)]
# Pentagon layout
_angles = [np.pi / 2 + i * 2 * np.pi / N for i in range(N)]
NODE_POS = {i: (0.9 * np.cos(a), 0.9 * np.sin(a)) for i, a in enumerate(_angles)}
NODE_R = 0.15

# Edges: x1→x2, x2→x3, x3→x1 (cycle 1), x3→x4, x4→x5, x5→x3 (cycle 2)
EDGES = [
    (0, 1), (1, 2), (2, 0),  # cycle 1: x1→x2→x3→x1
    (2, 3), (3, 4), (4, 2),  # cycle 2: x3→x4→x5→x3
]

# Greedy FVS: removing x3 (node 2) breaks both cycles
# Then all remaining edges are acyclic
REMOVAL_ORDER = [2]  # remove x3 to break both cycles

# Frame schedule:
# Phase 1: show full graph with red cycle edges (12 frames)
# Phase 2: for each removal — flash node, fade it out (8 frames per removal)
# Phase 3: show clean graph, HM score (10 frames hold)
# Total: ~30 frames

FRAMES_FULL = 12
FRAMES_PER_REMOVAL = 8
FRAMES_HOLD = 10

def build_schedule():
    schedule = []
    for f in range(FRAMES_FULL):
        schedule.append(("full", 0, f))
    for ri, node in enumerate(REMOVAL_ORDER):
        for f in range(FRAMES_PER_REMOVAL):
            schedule.append(("remove", ri, f))
    for f in range(FRAMES_HOLD):
        schedule.append(("hold", 0, f))
    return schedule

SCHEDULE = build_schedule()


def draw_node(ax, idx, faded=False):
    x, y = NODE_POS[idx]
    color = COL_FADED if faded else COL_NODE
    alpha = 0.3 if faded else 1.0
    shadow = Circle((x + 0.02, y - 0.02), radius=NODE_R,
                     facecolor="black", alpha=0.06 if faded else 0.12, zorder=0)
    ax.add_patch(shadow)
    node = Circle((x, y), radius=NODE_R, facecolor=color,
                   edgecolor="white", lw=2.5, zorder=10, alpha=alpha)
    ax.add_patch(node)
    ax.text(x, y, LABELS[idx], ha="center", va="center",
            fontsize=12, fontweight="bold", color="white",
            zorder=11, alpha=alpha)


def draw_edge(ax, src, dst, color=COL_EDGE, lw=2.0, alpha=0.85):
    x0, y0 = NODE_POS[src]
    x1, y1 = NODE_POS[dst]
    r = NODE_R
    dx, dy = x1 - x0, y1 - y0
    dist = np.hypot(dx, dy)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    sx, sy = x0 + ux * r, y0 + uy * r
    ex, ey = x1 - ux * r, y1 - uy * r
    arrow = FancyArrowPatch(
        (sx, sy), (ex, ey),
        connectionstyle="arc3,rad=0.12",
        arrowstyle="-|>,head_length=5,head_width=3",
        color=color, lw=lw, alpha=alpha, zorder=5,
    )
    ax.add_patch(arrow)


def generate_gif():
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=COL_BG)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.05, top=0.92)

    def update(frame_idx):
        ax.clear()
        ax.set_facecolor(COL_BG)
        ax.set_xlim(-1.6, 2.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

        phase, idx, pf = SCHEDULE[frame_idx]

        # Determine which nodes are removed so far
        if phase == "full":
            removed = set()
        elif phase == "remove":
            # Nodes removed before this one + current if past midpoint
            removed = set(REMOVAL_ORDER[:idx])
            if pf >= FRAMES_PER_REMOVAL // 2:
                removed.add(REMOVAL_ORDER[idx])
        else:
            removed = set(REMOVAL_ORDER)

        # Determine which edges are still active (both endpoints not removed)
        active_edges = [(s, d) for s, d in EDGES
                        if s not in removed and d not in removed]

        # Check if remaining edges have cycles (simple check: any cycle edges left?)
        # Cycle 1 edges: (0,1), (1,2), (2,0)
        # Cycle 2 edges: (2,3), (3,4), (4,2)
        cycle1_present = all((s, d) in active_edges for s, d in [(0, 1), (1, 2), (2, 0)])
        cycle2_present = all((s, d) in active_edges for s, d in [(2, 3), (3, 4), (4, 2)])
        cycle_edges = set()
        if cycle1_present:
            cycle_edges.update([(0, 1), (1, 2), (2, 0)])
        if cycle2_present:
            cycle_edges.update([(2, 3), (3, 4), (4, 2)])

        # Draw edges
        for s, d in EDGES:
            if s in removed or d in removed:
                # Faded edge
                draw_edge(ax, s, d, color=COL_FADED, lw=1.0, alpha=0.2)
            elif (s, d) in cycle_edges:
                draw_edge(ax, s, d, color=COL_RED, lw=2.5, alpha=0.85)
            else:
                draw_edge(ax, s, d, color=COL_EDGE, lw=2.0, alpha=0.85)

        # Draw nodes
        for i in range(N):
            faded = i in removed
            # Flash node being removed
            if phase == "remove" and i == REMOVAL_ORDER[idx] and pf < FRAMES_PER_REMOVAL // 2:
                if pf % 2 == 0:
                    draw_node(ax, i, faded=False)
                else:
                    draw_node(ax, i, faded=True)
            else:
                draw_node(ax, i, faded=faded)

        # HM score on the right
        n_remaining = N - len(removed)
        hm_str = f"{n_remaining}/{N}"

        x_anchor = 1.65
        ax.text(x_anchor, 0.5, "Houtman-Maks", ha="center", va="top",
                fontsize=10, fontweight="bold", color=COL_TEXT, zorder=20)
        ax.text(x_anchor, 0.15, hm_str, ha="center", va="top",
                fontsize=26, fontweight="bold", color=COL_NODE, zorder=20,
                fontfamily="monospace")

        # Status
        if phase == "full":
            ax.text(x_anchor, -0.3, "2 cycles\ndetected", ha="center",
                    fontsize=9, color=COL_RED, zorder=20)
        elif phase == "remove":
            ax.text(x_anchor, -0.3, f"Remove {LABELS[REMOVAL_ORDER[idx]]}\nto break cycles",
                    ha="center", fontsize=9, color=COL_SUBTEXT, zorder=20)
        else:
            ax.text(x_anchor, -0.3, "Acyclic", ha="center",
                    fontsize=9, fontweight="bold", color="#27ae60", zorder=20)

    anim = FuncAnimation(fig, update, frames=len(SCHEDULE), interval=250)
    out_path = OUT_DIR / "hm_algorithm.gif"
    anim.save(str(out_path), writer="pillow", dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB, {len(SCHEDULE)} frames)")


if __name__ == "__main__":
    generate_gif()
