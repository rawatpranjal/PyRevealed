"""
Generate animated preference-graph GIF for the PrefGraph homepage and LinkedIn.

Shows a single user making 10 sequential choices. Edges appear one by one;
when a cycle forms the cycle edges flash red. The Houtman-Maks score updates
live on the right side of the frame.

Usage:
    python3 tools/generate_consistency_gif.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.animation import FuncAnimation
from pathlib import Path
from prefgraph import MenuChoiceLog
from prefgraph.algorithms.abstract_choice import compute_menu_efficiency

# ---------------------------------------------------------------------------
# Output config
# ---------------------------------------------------------------------------
OUT_DIR = Path("docs/_static")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 150
FIG_W, FIG_H = 7.5, 4.5

# ---------------------------------------------------------------------------
# PrefGraph brand colours
# ---------------------------------------------------------------------------
COL_BG = "#fafafa"
COL_NODE = "#2563eb"
COL_NODE_LIGHT = "#3b82f6"
COL_EDGE = "#3b82f6"
COL_RED = "#e74c3c"
COL_SHADOW = "#00000026"       # 15% black
COL_TEXT = "#333333"
COL_SUBTEXT = "#666666"

# ---------------------------------------------------------------------------
# Graph layout — 6 nodes in a hexagonal arrangement
# ---------------------------------------------------------------------------
LABELS = list("ABCDEF")
N_ITEMS = len(LABELS)
# Hexagonal positions (centred on origin, radius ~0.9)
_hex_angles = [np.pi / 2 + i * 2 * np.pi / N_ITEMS for i in range(N_ITEMS)]
NODE_POS = {i: (0.9 * np.cos(a), 0.9 * np.sin(a)) for i, a in enumerate(_hex_angles)}
NODE_RADIUS = 0.16

# ---------------------------------------------------------------------------
# 10 choice observations — curated so violations start at obs 6
#
# Obs 1-5: consistent (acyclic preference graph)
#   1: A > B   2: B > C   3: A > D   4: D > E   5: E > F
# Obs 6: C > A  → creates cycle A>B>C>A  (first violation)
# Obs 7: F > D  → creates cycle D>E>F>D  (second violation)
# Obs 8: B > E  (consistent addition)
# Obs 9: C > D  (consistent addition)
# Obs 10: E > A → creates cycle A>D>E>A  (third violation)
# ---------------------------------------------------------------------------
OBSERVATIONS = [
    (frozenset({0, 1}), 0),   # A from {A,B}
    (frozenset({1, 2}), 1),   # B from {B,C}
    (frozenset({0, 3}), 0),   # A from {A,D}
    (frozenset({3, 4}), 3),   # D from {D,E}
    (frozenset({4, 5}), 4),   # E from {E,F}
    (frozenset({0, 2}), 2),   # C from {A,C}  ← CYCLE
    (frozenset({3, 5}), 5),   # F from {D,F}  ← CYCLE
    (frozenset({1, 4}), 1),   # B from {B,E}
    (frozenset({2, 3}), 2),   # C from {C,D}
    (frozenset({0, 4}), 4),   # E from {A,E}  ← CYCLE
]

# ---------------------------------------------------------------------------
# Pre-compute HM scores and edges at each step
# ---------------------------------------------------------------------------
def precompute():
    """Return list of dicts, one per observation, with HM results and edge info."""
    steps = []
    menus_so_far, choices_so_far = [], []
    prev_removed = set()

    for idx, (menu, choice) in enumerate(OBSERVATIONS):
        menus_so_far.append(menu)
        choices_so_far.append(choice)

        log = MenuChoiceLog(menus=list(menus_so_far), choices=list(choices_so_far))
        hm = compute_menu_efficiency(log)

        # The directed edge: choice was preferred over every other menu item
        unchosen = sorted(menu - {choice})
        # For the graph we draw one arrow: chosen → first unchosen (simplified)
        # (each obs is a binary menu so there's exactly one unchosen item)
        edge = (choice, unchosen[0])

        curr_removed = set(hm.removed_observations)
        # Did this observation trigger a NEW violation?
        is_violation = len(curr_removed) > len(prev_removed)
        prev_removed = curr_removed

        steps.append({
            "edge": edge,
            "hm_consistent": len(hm.remaining_observations),
            "hm_total": hm.num_total,
            "hm_ratio": hm.efficiency_index,
            "removed": curr_removed,
            "is_violation": is_violation,
            "obs_idx": idx,
        })
    return steps


STEPS = precompute()

# ---------------------------------------------------------------------------
# Frame schedule
#
# For each observation:
#   - 4 frames: edge appears (blue or red based on is_violation)
#   - if violation: 6 frames of red flash (3 cycles on/off)
#   - 2 frames: settle
# After all observations: 8 frames hold on final state (2s at 250ms)
# ---------------------------------------------------------------------------
FRAMES_APPEAR = 4
FRAMES_FLASH = 6       # 3 on/off cycles
FRAMES_SETTLE = 2
FRAMES_HOLD = 8

def build_frame_schedule():
    """Return list of (obs_index, phase, phase_frame) tuples."""
    schedule = []
    for obs_i, step in enumerate(STEPS):
        for f in range(FRAMES_APPEAR):
            schedule.append((obs_i, "appear", f))
        if step["is_violation"]:
            for f in range(FRAMES_FLASH):
                schedule.append((obs_i, "flash", f))
        for f in range(FRAMES_SETTLE):
            schedule.append((obs_i, "settle", f))
    # Hold final frame
    for f in range(FRAMES_HOLD):
        schedule.append((len(STEPS) - 1, "hold", f))
    return schedule


SCHEDULE = build_frame_schedule()

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_node(ax, item_id, alpha=1.0):
    """Draw a single node with drop shadow."""
    x, y = NODE_POS[item_id]
    r = NODE_RADIUS
    # Shadow
    shadow = Circle((x + 0.02, y - 0.02), radius=r,
                     facecolor="black", alpha=0.12, zorder=0,
                     transform=ax.transData)
    ax.add_patch(shadow)
    # Node
    node = Circle((x, y), radius=r, facecolor=COL_NODE,
                   edgecolor="white", lw=2.5, zorder=10, alpha=alpha)
    ax.add_patch(node)
    # Label
    ax.text(x, y, LABELS[item_id], ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", zorder=11)


def draw_edge(ax, src, dst, color=COL_EDGE, lw=2.5, alpha=0.85):
    """Draw a curved directed edge between two nodes."""
    x0, y0 = NODE_POS[src]
    x1, y1 = NODE_POS[dst]
    # Shorten the arrow so it doesn't overlap the node circles
    r = NODE_RADIUS
    dx, dy = x1 - x0, y1 - y0
    dist = np.hypot(dx, dy)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    # Start and end points pulled inward by node radius
    sx, sy = x0 + ux * r, y0 + uy * r
    ex, ey = x1 - ux * r, y1 - uy * r

    arrow = FancyArrowPatch(
        (sx, sy), (ex, ey),
        connectionstyle="arc3,rad=0.15",
        arrowstyle="-|>,head_length=6,head_width=4",
        color=color, lw=lw, alpha=alpha, zorder=5,
    )
    ax.add_patch(arrow)


def draw_hm_score(ax, consistent, total, ratio):
    """Draw the HM score on the right side of the figure."""
    # Position in axes coordinates (right side)
    x_anchor = 2.05
    y_top = 0.7

    ax.text(x_anchor, y_top, "Houtman-Maks", ha="center", va="top",
            fontsize=11, fontweight="bold", color=COL_TEXT, zorder=20)

    # Large fraction display
    score_str = f"{consistent}/{total}"
    ax.text(x_anchor, y_top - 0.35, score_str, ha="center", va="top",
            fontsize=28, fontweight="bold", color=COL_NODE, zorder=20,
            fontfamily="monospace")

    # Percentage below
    pct = f"{ratio * 100:.0f}% consistent"
    pct_color = COL_NODE if ratio >= 0.8 else (COL_RED if ratio < 0.6 else "#e67e22")
    ax.text(x_anchor, y_top - 0.75, pct, ha="center", va="top",
            fontsize=11, color=pct_color, zorder=20)

    # Observation counter
    ax.text(x_anchor, y_top - 1.1, f"Observation {total} of 10", ha="center",
            va="top", fontsize=9, color=COL_SUBTEXT, zorder=20)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def generate_gif():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=COL_BG)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    def update(frame_idx):
        ax.clear()
        ax.set_facecolor(COL_BG)
        # Graph area on the left, score on the right
        ax.set_xlim(-1.4, 2.8)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")

        obs_i, phase, phase_f = SCHEDULE[frame_idx]
        step = STEPS[obs_i]

        # Determine how many observations are fully visible
        n_visible = obs_i + 1  # current observation is being animated

        # Draw all nodes (always visible)
        for item_id in range(N_ITEMS):
            draw_node(ax, item_id)

        # Draw all edges up to current observation
        for ei in range(n_visible):
            s = STEPS[ei]
            src, dst = s["edge"]

            # Determine edge colour:
            # - if this edge's observation is in the "removed" set of the
            #   *current* step, it's a violation edge → red
            # - otherwise blue
            is_removed = ei in step["removed"]

            # Current observation during flash phase
            if ei == obs_i and phase == "flash":
                # Flash: alternate red/transparent every frame
                if phase_f % 2 == 0:
                    draw_edge(ax, src, dst, color=COL_RED, lw=3.5, alpha=1.0)
                else:
                    draw_edge(ax, src, dst, color=COL_RED, lw=2.5, alpha=0.3)
            elif is_removed:
                draw_edge(ax, src, dst, color=COL_RED, lw=2.5, alpha=0.85)
            else:
                draw_edge(ax, src, dst, color=COL_EDGE, lw=2.5, alpha=0.85)

        # Draw HM score
        draw_hm_score(ax, step["hm_consistent"], step["hm_total"], step["hm_ratio"])

        # Title
        ax.text(-0.0, 1.18, "Building a Preference Graph",
                ha="center", va="top", fontsize=14, fontweight="bold",
                color=COL_TEXT, zorder=20)

    anim = FuncAnimation(fig, update, frames=len(SCHEDULE), interval=250)
    out_path = OUT_DIR / "consistency.gif"
    anim.save(str(out_path), writer="pillow", dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}  ({out_path.stat().st_size / 1024:.0f} KB, {len(SCHEDULE)} frames)")


if __name__ == "__main__":
    generate_gif()
