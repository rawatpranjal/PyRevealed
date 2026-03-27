#!/usr/bin/env python3
"""Generate animated GIF visualizations for RTD documentation."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from pathlib import Path

from PIL import Image
import io

plt.switch_backend("Agg")

OUTPUT_DIR = Path(__file__).parent / "_static"
DPI = 100

# Durations for variable-speed algorithm GIFs
SLOW_MS = 3000   # text-change frames (phase transitions, new explanations)
FAST_MS = 600    # visual-only frames (edges appearing, nodes moving)


def _save_gif_variable_speed(fig, update_fn, total_frames, text_frames,
                              output_path, slow_ms=SLOW_MS, fast_ms=FAST_MS):
    """Save GIF with slow duration on text-change frames, fast on visual-only."""
    pil_frames = []
    durations = []
    for f in range(total_frames):
        update_fn(f)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor=fig.get_facecolor())
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        pil_frames.append(img.copy())
        durations.append(slow_ms if f in text_frames else fast_ms)
    pil_frames[0].save(
        output_path, save_all=True, append_images=pil_frames[1:],
        duration=durations, loop=0, optimize=False,
    )
PALETTE = {
    "bg": "#fafafa",
    "edge": "#4a4a4a",
    "node": "#5b8def",
    "node_text": "white",
    "highlight": "#e74c3c",
    "secondary": "#95a5a6",
    "rust": "#dea584",
    "python": "#4b8bbe",
    "accent": "#27ae60",
    "grid": "#e0e0e0",
}

# ---------------------------------------------------------------------------
# GIF 1A: Landing Page — Budget Choice
# Single-panel, 14-second story: detect violation → measure with CCEI
# ---------------------------------------------------------------------------
def generate_budget_hero():
    # Two shopping trips with a contradiction
    prices = np.array([[1.0, 2.0], [2.0, 1.0]])
    quants = np.array([[2.0, 3.0], [3.0, 2.0]])
    expenditures = np.sum(prices * quants, axis=1)  # [8, 8]
    colors = ["#5b8def", "#8e44ad"]
    CCEI_FINAL = 0.875
    TOTAL_FRAMES = 48

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), facecolor=PALETTE["bg"])
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.10, right=0.95)

    def _draw_arrow(ax, x0, y0, x1, y1, color, lw=1.5):
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length < 0.01:
            return
        shrink = 0.35
        sx, sy = x0 + shrink * dx / length, y0 + shrink * dy / length
        ex, ey = x1 - shrink * dx / length, y1 - shrink * dy / length
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0))

    def _draw_budget_lines(ax, e=1.0, ghost=False):
        """Draw budget lines at efficiency level e."""
        for t in range(2):
            p0, p1 = prices[t]
            budget = expenditures[t] * e
            x_int, y_int = budget / p0, budget / p1
            alpha = 0.15 if ghost else 0.5
            lw = 1.0 if ghost else 2.5
            ls = "--" if ghost else "-"
            ax.plot([0, x_int], [y_int, 0], color=colors[t], lw=lw, alpha=alpha, ls=ls)
            if not ghost:
                ax.fill_between([0, x_int], [y_int, 0], alpha=0.06, color=colors[t])

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-0.3, 9.0)
        ax.set_ylim(-0.3, 5.5)
        ax.set_xlabel("Apples (qty)", fontsize=11)
        ax.set_ylabel("Oranges (qty)", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.15, color=PALETTE["grid"])

        # Phase logic — gaps between text phases so the viewer can breathe
        if frame < 5:
            phase = "show"
        elif frame < 9:
            phase = "arrow1"
        elif frame < 11:
            phase = "gap1"       # arrows stay, text gone
        elif frame < 15:
            phase = "arrow2"
        elif frame < 17:
            phase = "gap2"       # both arrows, text gone
        elif frame < 21:
            phase = "contradiction"
        elif frame < 23:
            phase = "gap3"       # red arrows, text gone
        elif frame < 35:
            phase = "shrink"
        else:
            phase = "final"

        # Which arrows to show (carry forward through gaps)
        show_arrow1 = phase not in ("show",)
        show_arrow2 = phase not in ("show", "arrow1", "gap1")

        # Compute current efficiency level
        if phase == "shrink":
            progress = (frame - 23) / 11.0
            e = 1.0 - progress * (1.0 - CCEI_FINAL)
        elif phase == "final":
            e = CCEI_FINAL
        else:
            e = 1.0

        # Title
        if phase in ("show", "arrow1", "arrow2", "gap1", "gap2"):
            ax.set_title("Is this shopper rational?", fontsize=14,
                         fontweight="bold", pad=10, color=PALETTE["edge"])
        elif phase in ("contradiction", "gap3"):
            ax.set_title("Contradiction found", fontsize=14,
                         fontweight="bold", pad=10, color=PALETTE["highlight"])
        elif phase == "shrink":
            ax.set_title(f"Shrink budgets until contradiction disappears   e = {e:.2f}",
                         fontsize=13, fontweight="bold", pad=10, color="#e67e22")
        else:
            ax.set_title(f"CCEI = {CCEI_FINAL:.3f} — Afriat (1967)",
                         fontsize=14, fontweight="bold", pad=10, color=PALETTE["accent"])

        # Draw ghost lines if shrinking
        if phase in ("shrink", "final"):
            _draw_budget_lines(ax, e=1.0, ghost=True)

        # Draw budget lines at current efficiency
        _draw_budget_lines(ax, e=e)

        # Draw bundles (always)
        for t in range(2):
            ax.scatter(quants[t, 0], quants[t, 1], color=colors[t], s=140,
                       zorder=5, edgecolors="white", lw=2)
            label = f"Trip {t+1}"
            ax.annotate(label, (quants[t, 0] + 0.25, quants[t, 1] + 0.2),
                        fontsize=10, fontweight="bold", color=colors[t])

        # Preference arrows (persist through gap phases)
        arrow_color = PALETTE["highlight"] if phase in ("contradiction", "gap3") else PALETTE["edge"]
        if show_arrow1:
            _draw_arrow(ax, quants[0, 0], quants[0, 1],
                        quants[1, 0], quants[1, 1], color=arrow_color, lw=2.0)
        if show_arrow2:
            _draw_arrow(ax, quants[1, 0], quants[1, 1],
                        quants[0, 0], quants[0, 1], color=arrow_color, lw=2.0)

        # Phase-specific annotations
        if phase == "arrow1":
            ax.text(4.5, 4.8, "Trip 1 bundle was affordable\nat Trip 2 prices",
                    fontsize=10, color=colors[0], ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=colors[0], alpha=0.9))
        elif phase == "arrow2":
            ax.text(4.5, 4.8, "Trip 2 bundle was also affordable\nat Trip 1 prices",
                    fontsize=10, color=colors[1], ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=colors[1], alpha=0.9))
        elif phase == "contradiction":
            ax.text(4.5, 4.8, "Each bundle was affordable when\nthe other was picked — a cycle",
                    fontsize=10, color=PALETTE["highlight"], ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=PALETTE["highlight"], alpha=0.9))
        elif phase == "final":
            ax.text(6.5, 4.5, "88% rational",
                    fontsize=13, fontweight="bold", color=PALETTE["accent"], ha="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor=PALETTE["accent"], alpha=0.95))

    print("  Generating budget_hero.gif...")
    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=900)
    anim.save(OUTPUT_DIR / "budget_hero.gif", writer="pillow", dpi=DPI)
    plt.close(fig)

# ---------------------------------------------------------------------------
# GIF 1B: Landing Page — Menu Choice
# Single-panel, 14-second story: spot contradiction → count consistent → HM score
# ---------------------------------------------------------------------------
def generate_menu_hero():
    # Four menu observations with one contradiction (menus 1 vs 2)
    menus = [
        (["Laptop", "Tablet", "Phone"], "Laptop"),
        (["Laptop", "Tablet"], "Tablet"),
        (["Tablet", "Phone"], "Tablet"),
        (["Laptop", "Phone"], "Laptop"),
    ]
    item_colors = {
        "Laptop": "#5b8def",
        "Tablet": "#e67e22",
        "Phone": "#27ae60",
    }
    TOTAL_FRAMES = 46

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), facecolor=PALETTE["bg"])
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.95)

    def _draw_item_box(ax, x, y, text, chosen=False, color="#5b8def"):
        """Draw a labeled rectangle for a menu item."""
        w, h = 1.1, 0.4
        alpha = 1.0 if chosen else 0.25
        ec = color if chosen else PALETTE["secondary"]
        lw = 2.0 if chosen else 1.0
        rect = plt.Rectangle((x - w/2, y - h/2), w, h, facecolor=color,
                              alpha=alpha, edgecolor=ec, lw=lw, zorder=5,
                              clip_on=False)
        ax.add_patch(rect)
        text_color = "white" if chosen else PALETTE["secondary"]
        fw = "bold" if chosen else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                fontweight=fw, color=text_color, zorder=6, clip_on=False)

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-2.2, 7.5)
        ax.set_ylim(-4.2, 1.2)
        ax.axis("off")

        # Phase logic — gaps between text phases for breathing room
        if frame < 5:
            phase = "row1"
            n_rows = 1
        elif frame < 9:
            phase = "row2"
            n_rows = 2
        elif frame < 11:
            phase = "gap1"       # 2 rows visible, no annotation
            n_rows = 2
        elif frame < 16:
            phase = "contradiction"
            n_rows = 2
        elif frame < 18:
            phase = "gap2"       # 2 rows, no annotation
            n_rows = 2
        elif frame < 22:
            phase = "row3"
            n_rows = 3
        elif frame < 24:
            phase = "gap3"       # 3 rows, no annotation
            n_rows = 3
        elif frame < 28:
            phase = "row4"
            n_rows = 4
        elif frame < 30:
            phase = "gap4"       # 4 rows, no annotation
            n_rows = 4
        elif frame < 36:
            phase = "summary"
            n_rows = 4
        else:
            phase = "score"
            n_rows = 4

        # Title
        if phase in ("row1", "row2", "gap1"):
            ax.set_title("Is this user's clicking rational?", fontsize=14,
                         fontweight="bold", pad=10, color=PALETTE["edge"])
        elif phase in ("contradiction", "gap2"):
            ax.set_title("Contradiction found", fontsize=14,
                         fontweight="bold", pad=10, color=PALETTE["highlight"])
        elif phase in ("row3", "row4", "gap3", "gap4"):
            ax.set_title("Check remaining menus...", fontsize=14,
                         fontweight="bold", pad=10, color=PALETTE["edge"])
        elif phase == "summary":
            ax.set_title("3 of 4 choices are consistent", fontsize=14,
                         fontweight="bold", pad=10, color="#e67e22")
        else:
            ax.set_title("HM = 0.75 — Houtman & Maks (1985)", fontsize=14,
                         fontweight="bold", pad=10, color=PALETTE["accent"])

        # Draw menu rows
        for m_idx in range(n_rows):
            items, chosen = menus[m_idx]
            y_base = -m_idx * 0.9
            x_start = 1.2

            # Row label (with status indicator in summary/score phase)
            if phase in ("summary", "score"):
                is_bad = (m_idx == 0 or m_idx == 1)
                indicator = "\u2717 " if is_bad else "\u2713 "
                label_color = PALETTE["highlight"] if is_bad else PALETTE["accent"]
                ax.text(-0.8, y_base, f"{indicator}Menu {m_idx+1}:", fontsize=11,
                        va="center", ha="right", color=label_color, fontweight="bold")
            else:
                ax.text(-0.8, y_base, f"Menu {m_idx+1}:", fontsize=11,
                        va="center", ha="right", color=PALETTE["secondary"])

            # Highlight contradiction rows
            if phase == "contradiction" and m_idx in (0, 1):
                rect = plt.Rectangle((-1.4, y_base - 0.28), 8.2, 0.56,
                                     facecolor=PALETTE["highlight"], alpha=0.06,
                                     edgecolor="none", zorder=1, clip_on=False)
                ax.add_patch(rect)

            # Draw item boxes
            for i_idx, item in enumerate(items):
                x = x_start + i_idx * 1.4
                is_chosen = (item == chosen)
                _draw_item_box(ax, x, y_base, item, chosen=is_chosen,
                               color=item_colors[item])

            # "Picked:" label
            ax.text(x_start + len(items) * 1.4 + 0.2, y_base,
                    f"\u2192 {chosen}", fontsize=10, va="center",
                    color=item_colors[chosen], fontweight="bold")

        # Phase-specific annotations
        if phase == "row2":
            ax.text(3.0, -2.4,
                    "Wait \u2014 Laptop was available\nand they didn't pick it?",
                    fontsize=10, color="#e67e22",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#e67e22", alpha=0.9))
        elif phase == "contradiction":
            ax.text(3.0, -2.4,
                    "Menu 1: Laptop > Tablet\n"
                    "Menu 2: Tablet > Laptop\n"
                    "Both can't be true \u2014 a cycle",
                    fontsize=10, color=PALETTE["highlight"],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=PALETTE["highlight"], alpha=0.9))
        elif phase == "row3":
            ax.text(5.5, -2.7, "This one's fine",
                    fontsize=10, color=PALETTE["accent"],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=PALETTE["accent"], alpha=0.9))
        elif phase == "row4":
            ax.text(5.5, -3.5, "Consistent with Menu 1",
                    fontsize=10, color=PALETTE["accent"],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=PALETTE["accent"], alpha=0.9))
        elif phase == "score":
            ax.text(3.0, -3.8, "75% rational",
                    fontsize=13, fontweight="bold", color=PALETTE["accent"],
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor=PALETTE["accent"], alpha=0.95))

    print("  Generating menu_hero.gif...")
    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=900)
    anim.save(OUTPUT_DIR / "menu_hero.gif", writer="pillow", dpi=DPI)
    plt.close(fig)

def generate_floyd_warshall():
    """Floyd-Warshall educational GIF — slow, well-spaced, repo terminology."""
    fig, ax = plt.subplots(figsize=(8, 9), facecolor=PALETTE["bg"])
    plt.subplots_adjust(top=0.78, bottom=0.14, left=0.08, right=0.95)

    TOTAL_FRAMES = 120
    INTERVAL = 3200
    n = 5
    labels = [f"$x_{i+1}$" for i in range(n)]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    pos = np.column_stack([np.cos(angles), np.sin(angles)]) * 2.0
    NODE_R = 0.28

    # Direct R_0 edges
    direct = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)]
    direct_set = set(direct)

    # Precompute all FW states
    adj = np.zeros((n, n), dtype=bool)
    for i, j in direct:
        adj[i, j] = True
    states = [adj.copy()]
    focus_nodes = []
    for k in range(n):
        step_focus = []
        for i in range(n):
            for j in range(n):
                if adj[i, k] and adj[k, j] and not adj[i, j]:
                    adj[i, j] = True
                    step_focus.append((i, k, j))
        states.append(adj.copy())
        focus_nodes.append(step_focus)

    fig.text(0.04, 0.96, "Floyd-Warshall  Transitive Closure",
             fontsize=15, fontweight="bold", color=PALETTE["python"],
             family="monospace")
    phase_txt = fig.text(0.04, 0.90, "", fontsize=13, fontweight="bold", color="#333")
    desc_txt = fig.text(0.04, 0.845, "", fontsize=11, style="italic", color="#555",
                        linespacing=1.5)
    status_txt = fig.text(0.50, 0.055, "", fontsize=14, fontweight="bold",
                          ha="center", va="center", color=PALETTE["python"])
    counter_txt = fig.text(0.50, 0.02, "", fontsize=10, ha="center", va="center",
                           color="#666", family="monospace")

    def _arrow(ax, s, e, color, lw=1.5, style="-", alpha=1.0):
        dx, dy = e[0] - s[0], e[1] - s[1]
        ln = np.sqrt(dx**2 + dy**2)
        if ln < 0.01: return
        sh = NODE_R + 0.08
        ax.annotate("", xy=(e[0] - sh*dx/ln, e[1] - sh*dy/ln),
                     xytext=(s[0] + sh*dx/ln, s[1] + sh*dy/ln),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                     linestyle=style, shrinkA=0, shrinkB=0,
                                     alpha=alpha), zorder=5)

    def _node(ax, i, color=None, alpha=1.0):
        c = color or PALETTE["node"]
        ax.add_patch(plt.Circle(pos[i], NODE_R, color=c, alpha=alpha, zorder=10))
        ax.text(pos[i][0], pos[i][1], labels[i], ha="center", va="center",
                fontsize=12, color="white", fontweight="bold", zorder=11, alpha=alpha)

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("equal"); ax.axis("off")

        # Phase 1: Build R_0 (0-29)
        if frame <= 29:
            phase_txt.set_text("Step 1:  Build Direct Preference Graph  $R_0$")
            desc_txt.set_text(
                "From expenditure data:  $i \\; R_0 \\; j$  when  "
                "$E_{ii} \\geq E_{ij}$.\n"
                "Each edge = one revealed preference from observed prices.")
            status_txt.set_text("")
            for idx, (i, j) in enumerate(direct):
                appear = 3 + idx * 5
                if frame >= appear:
                    _arrow(ax, pos[i], pos[j], PALETTE["edge"], lw=2.0,
                           alpha=min((frame - appear) / 3, 1.0))
            for i in range(n): _node(ax, i)
            nv = sum(1 for idx in range(len(direct)) if frame >= 3 + idx * 5)
            counter_txt.set_text(f"$|R_0|$ edges: {nv} / {len(direct)}")

        # Phase 2: FW iterations (30-89), 12 frames per pivot
        elif frame <= 89:
            k = min((frame - 30) // 12, n - 1)
            sub = (frame - 30) % 12

            if sub <= 3:
                current = states[k]; active_triads = []
                phase_txt.set_text(
                    f"Step 2:  Pivot through  $x_{k+1}$   (k={k+1}/{n})")
                desc_txt.set_text(
                    f"For all pairs ($i$,$j$):  if $i \\; R \\; x_{k+1}$"
                    f"  and  $x_{k+1} \\; R \\; j$,  add  $i \\; R^* \\; j$.\n"
                    f"Scanning {n}\u00d7{n} pairs...")
            elif sub <= 7:
                current = states[k]; active_triads = focus_nodes[k]
                nn = len(focus_nodes[k])
                phase_txt.set_text(
                    f"Step 2:  Pivot through  $x_{k+1}$   (k={k+1}/{n})")
                if nn > 0:
                    desc_txt.set_text(
                        f"Found {nn} new transitive edge"
                        f"{'s' if nn != 1 else ''}!  "
                        "Adding to $R^*$.\n"
                        f"$i \\to x_{k+1} \\to j$  $\\Rightarrow$  "
                        "$i \\; R^* \\; j$  (dashed red).")
                else:
                    desc_txt.set_text(
                        f"No new edges through $x_{k+1}$.\n"
                        "Moving to next pivot.")
            else:
                current = states[k + 1]; active_triads = []
                te = int(current.sum())
                phase_txt.set_text(
                    f"Step 2:  Pivot through  $x_{k+1}$   (k={k+1}/{n})")
                desc_txt.set_text(
                    f"Pivot $x_{k+1}$ done.  $|R^*|$ = {te} edges.\n"
                    f"Total cost: O($T^3$) = O({n}\u00b3) = {n**3} ops.")

            hi_edges, new_edges = set(), set()
            for triad in active_triads:
                it, kn, jt = triad
                hi_edges.add((it, kn)); hi_edges.add((kn, jt))
                new_edges.add((it, jt))

            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if current[i, j]:
                        isd = (i, j) in direct_set
                        if (i, j) in new_edges:
                            _arrow(ax, pos[i], pos[j], PALETTE["highlight"],
                                   lw=2.5, style="--")
                        elif (i, j) in hi_edges:
                            _arrow(ax, pos[i], pos[j], PALETTE["python"],
                                   lw=3.0, style="-" if isd else "--")
                        else:
                            _arrow(ax, pos[i], pos[j],
                                   PALETTE["edge"] if isd else PALETTE["secondary"],
                                   lw=2.0 if isd else 1.5,
                                   style="-" if isd else "--")
                    if (sub > 3 and (i, j) in new_edges
                            and states[k + 1][i, j]):
                        _arrow(ax, pos[i], pos[j], PALETTE["highlight"],
                               lw=2.5, style="--")

            for i in range(n):
                _node(ax, i, PALETTE["python"] if i == k else None)
            tn = int(current.sum()) + len(new_edges)
            counter_txt.set_text(
                f"Pivot: k={k+1}/{n}  |  $|R^*|$: {tn}  |  O(T\u00b3)")
            status_txt.set_text("")

        # Phase 3: Complete (90-119)
        else:
            current = states[-1]; te = int(current.sum())
            phase_txt.set_text("Result:  Transitive Closure  $R^*$  Complete")
            desc_txt.set_text(
                f"All indirect preferences inferred.  "
                f"$|R^*|$ = {te} edges.\n"
                "Cost: O($T^3$).  "
                "For T=10,000 this is 10\u00b9\u00b2 operations.")
            for i in range(n):
                for j in range(n):
                    if i != j and current[i, j]:
                        isd = (i, j) in direct_set
                        _arrow(ax, pos[i], pos[j],
                               PALETTE["edge"] if isd else PALETTE["secondary"],
                               lw=2.0 if isd else 1.2,
                               style="-" if isd else "--", alpha=0.7)
            for i in range(n): _node(ax, i)
            a = min((frame - 90) / 5, 1.0)
            status_txt.set_text(
                f"$R^*$ = {te} edges  (from {len(direct)} direct)")
            status_txt.set_alpha(a)
            counter_txt.set_text(
                "Now check: any $i R^* j$ with $j P_0 i$?  "
                "\u2192 GARP violation")

    # Text-change frames: phase starts + each FW pivot sub-phase boundary
    text_frames = {0, 90}
    for k in range(n):
        base = 30 + k * 12
        text_frames.update({base, base + 4, base + 8})

    print("  Generating floyd_warshall.gif...")
    _save_gif_variable_speed(fig, update, TOTAL_FRAMES, text_frames,
                              OUTPUT_DIR / "floyd_warshall.gif")
    plt.close(fig)


# ---------------------------------------------------------------------------
# GIF 3: Tarjan's SCC (O(T^2))
# ---------------------------------------------------------------------------
def generate_scc_tarjan():
    """SCC-based GARP check — slow, well-spaced, repo terminology."""
    fig, ax = plt.subplots(figsize=(8, 9), facecolor=PALETTE["bg"])
    plt.subplots_adjust(top=0.78, bottom=0.14, left=0.08, right=0.95)

    TOTAL_FRAMES = 120
    INTERVAL = 3200
    n = 6
    labels = ["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$", "$x_6$"]
    pos = np.array([
        [-1.2, 1.8], [1.2, 1.8], [0.0, 0.0],
        [-1.8, -2.0], [1.8, -2.0], [0.0, -3.2],
    ])
    NODE_R = 0.28
    r0_edges = [(0,1),(1,2),(2,0),(2,3),(2,4),(3,4),(4,3),(3,5),(4,5)]
    p0_edges = {(2, 0)}
    scc_a = {0, 1, 2}
    scc_b = {3, 4}

    fig.text(0.04, 0.96, "Tarjan's SCC  (GARP in O(T\u00b2))",
             fontsize=15, fontweight="bold", color=PALETTE["rust"],
             family="monospace")
    phase_txt = fig.text(0.04, 0.90, "", fontsize=13, fontweight="bold", color="#333")
    desc_txt = fig.text(0.04, 0.845, "", fontsize=11, style="italic", color="#555",
                        linespacing=1.5)
    status_txt = fig.text(0.50, 0.055, "", fontsize=14, fontweight="bold",
                          ha="center", va="center", color=PALETTE["rust"])
    counter_txt = fig.text(0.50, 0.02, "", fontsize=10, ha="center", va="center",
                           color="#666", family="monospace")

    def _arrow(ax, i, j, color, lw=2.0, style="-", alpha=1.0):
        dx, dy = pos[j][0]-pos[i][0], pos[j][1]-pos[i][1]
        ln = np.sqrt(dx**2+dy**2)
        if ln < 0.01: return
        sh = NODE_R+0.08
        ax.annotate("", xy=(pos[j][0]-sh*dx/ln, pos[j][1]-sh*dy/ln),
                     xytext=(pos[i][0]+sh*dx/ln, pos[i][1]+sh*dy/ln),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                     ls=style, shrinkA=0, shrinkB=0,
                                     alpha=alpha), zorder=5)

    def _node(ax, i, color=None, alpha=1.0):
        c = color or PALETTE["node"]
        ax.add_patch(plt.Circle(pos[i], NODE_R, color=c, alpha=alpha, zorder=10))
        ax.text(pos[i][0], pos[i][1], labels[i], ha="center", va="center",
                fontsize=12, color="white", fontweight="bold", zorder=11, alpha=alpha)

    def _scc_bubble(ax, nodes, color, label=None):
        xs = [pos[i][0] for i in nodes]
        ys = [pos[i][1] for i in nodes]
        cx, cy = np.mean(xs), np.mean(ys)
        r = max(max(abs(x-cx) for x in xs), max(abs(y-cy) for y in ys)) + 0.7
        ax.add_patch(plt.Circle((cx, cy), r, fill=True, color=color,
                                alpha=0.12, zorder=1))
        if label:
            ax.text(cx, cy+r+0.25, label, ha="center", fontsize=9,
                    fontweight="bold", color=color, zorder=2)

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-4.2, 3.2)
        ax.set_aspect("equal"); ax.axis("off")

        # Phase 1: Build R_0 and P_0 (0-29)
        if frame <= 29:
            phase_txt.set_text("Step 1:  Build $R_0$ and $P_0$ from Expenditure Data")
            desc_txt.set_text(
                "$R_0$: $i \\; R_0 \\; j$  when $E_{ii} \\geq E_{ij}$"
                "  (weak preference).\n"
                "$P_0$: $i \\; P_0 \\; j$  when $E_{ii} > E_{ij}$"
                "  (strict preference).")
            status_txt.set_text("")
            for idx, (u, v) in enumerate(r0_edges):
                appear = 2 + idx * 3
                if frame >= appear:
                    a = min((frame-appear)/2, 1.0)
                    is_s = (u, v) in p0_edges
                    _arrow(ax, u, v,
                           PALETTE["highlight"] if is_s else PALETTE["edge"],
                           lw=2.5 if is_s else 1.5, alpha=a)
            for i in range(n): _node(ax, i)
            nv = sum(1 for idx in range(len(r0_edges)) if frame >= 2+idx*3)
            counter_txt.set_text(
                f"$|R_0|$: {nv}  |  $|P_0|$: {1 if nv >= 3 else 0}  "
                "(red = strict $P_0$)")

        # Phase 2: SCC decomposition (30-59)
        elif frame <= 59:
            phase_txt.set_text("Step 2:  Tarjan's SCC Decomposition  O($T^2$)")
            status_txt.set_text("")
            if frame <= 42:
                desc_txt.set_text(
                    "Group nodes into Strongly Connected Components.\n"
                    "Nodes that can reach each other via $R_0$ form an SCC.")
                counter_txt.set_text(
                    "SCC A = {$x_1$,$x_2$,$x_3$}  |  "
                    "SCC B = {$x_4$,$x_5$}  |  {$x_6$} singleton")
            else:
                desc_txt.set_text(
                    "Within an SCC, all nodes are mutually reachable.\n"
                    "Cross-SCC edges form a DAG.  No closure needed!")
                counter_txt.set_text("3 components found in O($T^2$)")
            _scc_bubble(ax, [0,1,2], PALETTE["rust"], "SCC A")
            if frame >= 36:
                _scc_bubble(ax, [3,4], PALETTE["accent"], "SCC B")
            for u, v in r0_edges:
                is_s = (u, v) in p0_edges
                if u in scc_a and v in scc_a:
                    c = PALETTE["highlight"] if is_s else PALETTE["rust"]
                elif u in scc_b and v in scc_b:
                    c = PALETTE["accent"]
                else:
                    c = PALETTE["secondary"]
                _arrow(ax, u, v, c, lw=2.0 if is_s else 1.5)
            for i in range(n):
                if i in scc_a: _node(ax, i, PALETTE["rust"])
                elif i in scc_b: _node(ax, i, PALETTE["accent"])
                else: _node(ax, i, PALETTE["secondary"])

        # Phase 3: Screen SCCs for P_0 (60-89)
        elif frame <= 89:
            phase_txt.set_text("Step 3:  Screen Each SCC for Strict Arc  $P_0$")
            status_txt.set_text("")
            _scc_bubble(ax, [0,1,2], PALETTE["rust"], "SCC A")
            _scc_bubble(ax, [3,4], PALETTE["accent"], "SCC B")
            if frame <= 72:
                desc_txt.set_text(
                    "GARP Theorem:  violation $\\Leftrightarrow$ some SCC contains a $P_0$ arc.\n"
                    "Scanning SCC A edges for strict preferences...")
                counter_txt.set_text("Checking SCC A for strict arcs...")
                for u, v in r0_edges:
                    if u in scc_a and v in scc_a:
                        is_s = (u, v) in p0_edges
                        if is_s:
                            pulse = 0.5+0.5*abs(np.sin(frame*0.4))
                            _arrow(ax, u, v, PALETTE["highlight"], lw=3.5,
                                   alpha=pulse)
                        else:
                            _arrow(ax, u, v, PALETTE["rust"], lw=1.5)
                    else:
                        _arrow(ax, u, v, PALETTE["secondary"], lw=1.0, alpha=0.4)
            else:
                desc_txt.set_text(
                    "Found!  $x_3 \\; P_0 \\; x_1$  is a strict arc"
                    " inside SCC A.\n"
                    "$E_{3,3} > E_{3,1}$:  $x_1$ strictly cheaper, yet in same SCC.")
                counter_txt.set_text(
                    "\u2716 GARP VIOLATED  |  "
                    "$P_0$ arc ($x_3 \\to x_1$) inside SCC A")
                for u, v in r0_edges:
                    if (u, v) in p0_edges:
                        pulse = 0.5+0.5*abs(np.sin(frame*0.4))
                        _arrow(ax, u, v, PALETTE["highlight"], lw=4.0, alpha=pulse)
                    elif u in scc_a and v in scc_a:
                        _arrow(ax, u, v, PALETTE["rust"], lw=1.5)
                    else:
                        _arrow(ax, u, v, PALETTE["secondary"], lw=1.0, alpha=0.3)
                mx = (pos[2][0]+pos[0][0])/2
                my = (pos[2][1]+pos[0][1])/2+0.5
                ax.text(mx, my, "\u2716", color=PALETTE["highlight"],
                        fontsize=40, ha="center", va="center", zorder=0, alpha=0.7)
            for i in range(n):
                if i in scc_a: _node(ax, i, PALETTE["rust"])
                elif i in scc_b: _node(ax, i, PALETTE["accent"])
                else: _node(ax, i, PALETTE["secondary"])

        # Phase 4: Result (90-119)
        else:
            phase_txt.set_text("Result:  GARP Violation Detected in O($T^2$)")
            desc_txt.set_text(
                "No transitive closure needed!  SCC + $P_0$ scan suffices.\n"
                "Talla Nobibon et al. (2015): O($T^2$) vs O($T^3$) FW.")
            _scc_bubble(ax, [0,1,2], PALETTE["rust"], "SCC A  \u2716")
            _scc_bubble(ax, [3,4], PALETTE["accent"], "SCC B  \u2714")
            for u, v in r0_edges:
                if (u, v) in p0_edges:
                    _arrow(ax, u, v, PALETTE["highlight"], lw=3.5)
                elif u in scc_a and v in scc_a:
                    _arrow(ax, u, v, PALETTE["rust"], lw=1.5, alpha=0.6)
                elif u in scc_b and v in scc_b:
                    _arrow(ax, u, v, PALETTE["accent"], lw=1.5, alpha=0.6)
                else:
                    _arrow(ax, u, v, PALETTE["secondary"], lw=1.0, alpha=0.3)
            for i in range(n):
                if i in scc_a: _node(ax, i, PALETTE["rust"])
                elif i in scc_b: _node(ax, i, PALETTE["accent"])
                else: _node(ax, i, PALETTE["secondary"])
            a = min((frame-90)/5, 1.0)
            status_txt.set_text("GARP violated:  $P_0$ arc inside SCC A")
            status_txt.set_alpha(a)
            counter_txt.set_text(
                "O(T\u00b2) SCC check  vs  O(T\u00b3) Floyd-Warshall  "
                "\u2014  1000\u00d7 faster at T=10,000")

    # Text-change frames: phase starts + sub-phase text transitions
    text_frames = {0, 30, 43, 60, 73, 90}

    print("  Generating scc_tarjan.gif...")
    _save_gif_variable_speed(fig, update, TOTAL_FRAMES, text_frames,
                              OUTPUT_DIR / "scc_tarjan.gif")
    plt.close(fig)



def generate_power_analysis():
    """Random CCEI histogram building up, then observed score drops in."""
    np.random.seed(42)
    # Random CCEI scores (Bronars-style: uniform on budget, compute CCEI)
    random_scores = np.random.beta(2, 8, size=500)
    random_scores = np.clip(random_scores, 0, 1)
    observed_ccei = 0.89  # typical empirical value

    n_build = 25  # frames building histogram
    n_hold_hist = 3
    n_drop = 5  # frames for observed line
    n_hold_final = 5
    total = n_build + n_hold_hist + n_drop + n_hold_final
    bins = np.linspace(0, 1, 30)

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=PALETTE["bg"])

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])

        if frame < n_build:
            # incrementally add samples
            n_samples = int((frame + 1) / n_build * len(random_scores))
            data = random_scores[:n_samples]
        else:
            data = random_scores

        ax.hist(data, bins=bins, color=PALETTE["secondary"], edgecolor="white", alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 80)
        ax.set_xlabel("CCEI Score", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, color=PALETTE["grid"])

        # After histogram is built, drop in the observed line
        drop_start = n_build + n_hold_hist
        if frame >= drop_start:
            ax.axvline(observed_ccei, color=PALETTE["highlight"], lw=2.5, linestyle="--")
            ax.annotate(
                f"Observed\nCCEI = {observed_ccei}",
                xy=(observed_ccei, 70), xytext=(observed_ccei + 0.06, 70),
                fontsize=10, fontweight="bold", color=PALETTE["highlight"],
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["highlight"]),
            )

        ax.set_title("Power Analysis: Observed vs Random", fontsize=12, fontweight="bold")
        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=total, interval=350)
    anim.save(OUTPUT_DIR / "power_analysis.gif", writer="pillow", dpi=DPI)
    plt.close(fig)
    print("  power_analysis.gif")


# ---------------------------------------------------------------------------
# GIF 5: Batch Engine Throughput
# ---------------------------------------------------------------------------
def generate_engine_throughput():
    """Rust vs Python racing bars showing throughput difference."""
    rust_rate = 49000  # agents/sec (GARP only)
    python_rate = 3200  # agents/sec estimate
    target = 100000  # total agents to process

    n_frames = 40
    hold_frames = 5

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PALETTE["bg"])

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])

        if frame < n_frames:
            t = (frame + 1) / n_frames
            rust_done = min(int(t * rust_rate * (target / rust_rate)), target)
            python_done = min(int(t * python_rate * (target / rust_rate)), target)
        else:
            rust_done = target
            python_done = min(int(python_rate * (target / rust_rate)), target)

        bars = ax.barh(
            ["Python", "Rust"],
            [python_done, rust_done],
            color=[PALETTE["python"], PALETTE["rust"]],
            edgecolor="white",
            height=0.5,
        )

        ax.set_xlim(0, target * 1.15)
        ax.set_xlabel("Agents Processed", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3, color=PALETTE["grid"])

        for bar, val in zip(bars, [python_done, rust_done]):
            ax.text(
                bar.get_width() + target * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=10, fontweight="bold",
            )

        ax.set_title("Engine Throughput: Rust vs Python", fontsize=12, fontweight="bold")

        if frame >= n_frames:
            speedup = rust_rate / python_rate
            ax.text(
                target * 0.55, -0.6,
                f"{speedup:.0f}x faster",
                fontsize=14, fontweight="bold", color=PALETTE["rust"],
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE["rust"]),
            )

        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=n_frames + hold_frames, interval=150)
    anim.save(OUTPUT_DIR / "engine_throughput.gif", writer="pillow", dpi=DPI)
    plt.close(fig)
    print("  engine_throughput.gif")


# ---------------------------------------------------------------------------
# GIF 6: Attention Decay
# ---------------------------------------------------------------------------
def generate_attention_decay():
    """Menu items fading as position-based attention drops off."""
    items = ["Item A", "Item B", "Item C", "Item D", "Item E", "Item F", "Item G", "Item H"]
    n_items = len(items)
    # position-based attention probabilities (higher position = more attention)
    attention = np.array([0.95, 0.88, 0.75, 0.60, 0.42, 0.28, 0.15, 0.08])
    threshold = 0.5  # consideration set threshold

    n_fade = 15
    n_hold = 5
    total = n_fade + n_hold

    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=PALETTE["bg"])

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])

        progress = min(frame / max(n_fade - 1, 1), 1.0)

        y_pos = np.arange(n_items)[::-1]
        current_alpha = 1.0 - progress * (1.0 - attention)

        for i in range(n_items):
            alpha = current_alpha[i]
            in_consideration = attention[i] >= threshold
            color = PALETTE["node"] if in_consideration else PALETTE["secondary"]
            ax.barh(
                y_pos[i], attention[i], height=0.6,
                color=color, alpha=max(alpha, 0.15), edgecolor="white", lw=1,
            )
            ax.text(
                -0.08, y_pos[i], items[i],
                va="center", ha="right", fontsize=10,
                alpha=max(alpha, 0.3),
            )
            ax.text(
                attention[i] + 0.02, y_pos[i],
                f"{attention[i]:.0%}",
                va="center", fontsize=9, alpha=max(alpha, 0.3),
            )

        # consideration set bracket
        if frame >= n_fade - 3:
            n_considered = int(np.sum(attention >= threshold))
            ax.axvline(threshold, color=PALETTE["highlight"], lw=1.5, linestyle=":", alpha=0.7)
            ax.text(
                threshold, n_items - 0.3, " threshold",
                fontsize=8, color=PALETTE["highlight"], va="bottom",
            )
            ax.annotate(
                f"Consideration set: {n_considered}/{n_items}",
                xy=(0.6, -0.8), fontsize=10, fontweight="bold",
                color=PALETTE["accent"],
            )

        ax.set_xlim(-0.1, 1.15)
        ax.set_ylim(-1.2, n_items)
        ax.set_xlabel("Attention Probability", fontsize=10)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3, color=PALETTE["grid"])
        ax.set_title("Position-Based Attention Decay", fontsize=12, fontweight="bold")
        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=total, interval=350)
    anim.save(OUTPUT_DIR / "attention_decay.gif", writer="pillow", dpi=DPI)
    plt.close(fig)
    print("  attention_decay.gif")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GIF: CCEI (Afriat Efficiency Index) Pedagogical
# ---------------------------------------------------------------------------
def generate_ccei_algorithm():
    """CCEI educational GIF — slow, well-spaced, no text overlap."""
    fig, ax = plt.subplots(figsize=(8, 9), facecolor=PALETTE["bg"])
    plt.subplots_adjust(top=0.78, bottom=0.16, left=0.10, right=0.95)

    TOTAL_FRAMES = 130
    INTERVAL = 3200  # ms — slow cadence

    # --- data: 2 goods, 2 observations with a GARP violation ---
    prices = np.array([[1.0, 2.0], [2.0, 1.0]])
    quants = np.array([[3.0, 2.5], [2.5, 3.0]])
    expends = np.sum(prices * quants, axis=1)  # [8.0, 8.0]
    CCEI_STAR = expends[0] / (prices[0] @ quants[1])  # critical e
    colors_obs = ["#5b8def", "#8e44ad"]
    labels_obs = ["$x_1$", "$x_2$"]
    # expenditure matrix E[i,j] = p_i · x_j
    E_matrix = prices @ quants.T  # 2×2

    # --- fixed title (never changes, never overlaps) ---
    fig.text(
        0.04, 0.96,
        "CCEI  (Critical Cost Efficiency Index)",
        fontsize=15, fontweight="bold", color=PALETTE["accent"],
        family="monospace",
    )

    # --- persistent text objects for phase / description / score ---
    phase_txt = fig.text(0.04, 0.90, "", fontsize=13, fontweight="bold", color="#333")
    desc_txt = fig.text(0.04, 0.845, "", fontsize=11, style="italic", color="#555",
                        linespacing=1.5)
    score_txt = fig.text(0.50, 0.06, "", fontsize=16, fontweight="bold",
                         ha="center", va="center", color=PALETTE["accent"])
    meter_txt = fig.text(0.50, 0.025, "", fontsize=10, ha="center", va="center",
                         color="#666", family="monospace")

    def _draw_budget(ax, idx, e, alpha_line=1.0, alpha_fill=0.08, lw=3):
        """Draw a single budget line + shaded affordable region."""
        p = prices[idx]
        E = expends[idx] * e
        x_int = E / p[0]
        y_int = E / p[1]
        ax.plot([0, x_int], [y_int, 0], color=colors_obs[idx], lw=lw,
                alpha=alpha_line, zorder=3)
        ax.fill_between([0, x_int], [y_int, 0], alpha=alpha_fill,
                        color=colors_obs[idx], zorder=1)

    def _draw_dot(ax, idx, show_data=False):
        """Draw a choice dot with label offset."""
        x, y = quants[idx]
        ax.scatter(x, y, color=colors_obs[idx], s=180, zorder=6,
                   edgecolors="white", lw=2.5)
        # offset label away from the other dot to avoid overlap
        ox = 0.35 if idx == 0 else -0.85
        oy = -0.45 if idx == 0 else 0.35
        ax.text(x + ox, y + oy, labels_obs[idx], fontsize=13,
                fontweight="bold", color=colors_obs[idx], zorder=7)
        if show_data:
            # show price and quantity vectors near the dot
            p = prices[idx]
            q = quants[idx]
            data_ox = 1.2 if idx == 0 else -2.8
            data_oy = -0.9 if idx == 0 else 0.9
            ax.text(
                x + data_ox, y + data_oy,
                f"$p_{idx+1}$=({p[0]:.0f},{p[1]:.0f})  "
                f"$x_{idx+1}$=({q[0]:.0f},{q[1]:.1f})\n"
                f"$E_{{{idx+1},{idx+1}}}$ = $p_{idx+1} \\cdot x_{idx+1}$ = {expends[idx]:.0f}",
                fontsize=8, color=colors_obs[idx], family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=colors_obs[idx], alpha=0.8, lw=0.8),
                zorder=8, va="center",
            )

    def _draw_pref_arrow(ax, src, dst, color, lw=2, alpha=1.0, label=None):
        """Curved preference arrow between two bundles."""
        xs, ys = quants[src]
        xd, yd = quants[dst]
        dx, dy = xd - xs, yd - ys
        length = np.sqrt(dx**2 + dy**2)
        shrink = 0.32
        sx = xs + shrink * dx / length
        sy = ys + shrink * dy / length
        ex = xd - shrink * dx / length
        ey = yd - shrink * dy / length
        ax.annotate(
            "", xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                connectionstyle="arc3,rad=0.25", shrinkA=0, shrinkB=0,
            ),
            zorder=5,
        )
        if label:
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            # offset perpendicular to arrow
            perp_x, perp_y = -dy / length, dx / length
            ax.text(
                mx + perp_x * 0.55, my + perp_y * 0.55, label,
                fontsize=9, ha="center", va="center", color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85),
                zorder=8,
            )

    def _draw_meter(e_val, is_done=False):
        """Update bottom efficiency meter text."""
        bar_len = 30
        filled = int(round(e_val * bar_len))
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
        meter_txt.set_text(f"e = {e_val:.3f}   [{bar}]")
        if is_done:
            score_txt.set_text(f"CCEI = {e_val:.3f}")
        else:
            score_txt.set_text("")

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-0.3, 9.0)
        ax.set_ylim(-0.3, 9.0)
        ax.set_xlabel("Good 1", fontsize=12, labelpad=8)
        ax.set_ylabel("Good 2", fontsize=12, labelpad=8)
        ax.grid(True, alpha=0.12, color=PALETTE["grid"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["grid"])

        # ============================================================
        # Phase 1: Setup (frames 0-24) — draw budget lines + dots
        # ============================================================
        if frame <= 24:
            phase_txt.set_text("Step 1:  Two Observations  (T = 2 goods)")
            desc_txt.set_text(
                "Prices $p_t$ and quantities $x_t$ for T=2 observations.\n"
                "Budget line at $t$:  $p_t \\cdot x = E_{tt}$  (own expenditure)."
            )
            meter_txt.set_text("")
            score_txt.set_text("")

            # budget line 1 appears at frame 3
            if frame >= 3:
                prog1 = min((frame - 3) / 5, 1.0)
                _draw_budget(ax, 0, 1.0, alpha_line=prog1)
            # dot 1 with data appears at frame 8
            if frame >= 8:
                _draw_dot(ax, 0, show_data=(frame >= 12))
            # budget line 2 appears at frame 14
            if frame >= 14:
                prog2 = min((frame - 14) / 5, 1.0)
                _draw_budget(ax, 1, 1.0, alpha_line=prog2)
            # dot 2 with data appears at frame 19
            if frame >= 19:
                _draw_dot(ax, 1, show_data=(frame >= 22))

        # ============================================================
        # Phase 2: Reveal preference cycle (frames 25-49)
        # ============================================================
        elif frame <= 49:
            _draw_budget(ax, 0, 1.0)
            _draw_budget(ax, 1, 1.0)
            _draw_dot(ax, 0)
            _draw_dot(ax, 1)
            meter_txt.set_text("")
            score_txt.set_text("")

            e12 = E_matrix[0, 1]  # p1·x2
            e21 = E_matrix[1, 0]  # p2·x1

            if frame <= 36:
                phase_txt.set_text("Step 2:  Revealed Preference  $R_0$")
                desc_txt.set_text(
                    f"$E_{{1,1}}$ = {expends[0]:.0f}  $\\geq$  $E_{{1,2}}$ = $p_1 \\cdot x_2$ = {e12:.0f}."
                    f"  $x_2$ was affordable!\n"
                    f"$x_1$ chosen over $x_2$  $\\Rightarrow$   $x_1 \\; R_0 \\; x_2$  (revealed preferred)."
                )
                prog = min((frame - 25) / 4, 1.0)
                _draw_pref_arrow(ax, 0, 1, PALETTE["node"], lw=2.5,
                                 alpha=prog, label="$x_1 R_0 x_2$")
            else:
                desc_txt.set_text(
                    f"$E_{{2,2}}$ = {expends[1]:.0f}  $\\geq$  $E_{{2,1}}$ = $p_2 \\cdot x_1$ = {e21:.0f}."
                    f"  $x_1$ was affordable!\n"
                    f"$x_2$ chosen over $x_1$  $\\Rightarrow$   $x_2 \\; R_0 \\; x_1$."
                )
                _draw_pref_arrow(ax, 0, 1, PALETTE["node"], lw=2.5,
                                 label="$x_1 R_0 x_2$")
                prog = min((frame - 37) / 4, 1.0)
                _draw_pref_arrow(ax, 1, 0, PALETTE["node"], lw=2.5,
                                 alpha=prog, label="$x_2 R_0 x_1$")
                if frame >= 43:
                    phase_txt.set_text("Step 2:  GARP Violation!")
                    desc_txt.set_text(
                        "$x_1 \\; R_0 \\; x_2$  AND  $x_2 \\; R_0 \\; x_1$"
                        "  forms a 2-cycle.\n"
                        "GARP requires no such cycles.  This data is inconsistent."
                    )
                    _draw_pref_arrow(ax, 0, 1, PALETTE["highlight"], lw=3)
                    _draw_pref_arrow(ax, 1, 0, PALETTE["highlight"], lw=3)
                    # red X in center
                    mx = (quants[0, 0] + quants[1, 0]) / 2
                    my = (quants[0, 1] + quants[1, 1]) / 2
                    ax.text(mx, my + 0.8, "\u2716 GARP Violated", fontsize=13,
                            fontweight="bold", color=PALETTE["highlight"],
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", edgecolor=PALETTE["highlight"],
                                      lw=1.5, alpha=0.9),
                            zorder=10)

        # ============================================================
        # Phase 3: Introduce shrinking idea (frames 50-64)
        # ============================================================
        elif frame <= 64:
            phase_txt.set_text("Step 3:  Deflate Budgets by  $e$")
            desc_txt.set_text(
                "Replace $E_{tt}$ with $e \\cdot E_{tt}$.  "
                "Shrink each budget inward.\n"
                "CCEI = sup$\\{e : \\text{deflated data satisfies GARP}\\}$."
            )
            meter_txt.set_text("e = 1.000   (no deflation yet)")
            score_txt.set_text("")

            # draw original budgets as ghosts
            _draw_budget(ax, 0, 1.0, alpha_line=0.25, alpha_fill=0.03, lw=1.5)
            _draw_budget(ax, 1, 1.0, alpha_line=0.25, alpha_fill=0.03, lw=1.5)
            # draw slightly shrunken budgets to preview
            preview_e = 1.0 - 0.04 * min((frame - 50) / 14, 1.0)
            _draw_budget(ax, 0, preview_e, alpha_line=0.8)
            _draw_budget(ax, 1, preview_e, alpha_line=0.8)
            _draw_dot(ax, 0)
            _draw_dot(ax, 1)

            # show inward arrows on budget lines
            if frame >= 55:
                for idx in range(2):
                    p = prices[idx]
                    E = expends[idx]
                    mx = (E / p[0]) / 2
                    my = (E / p[1]) / 2
                    # arrow pointing inward
                    dx, dy = -p[0], -p[1]
                    length = np.sqrt(dx**2 + dy**2)
                    ax.annotate(
                        "", xy=(mx + 0.4 * dx / length, my + 0.4 * dy / length),
                        xytext=(mx, my),
                        arrowprops=dict(arrowstyle="-|>", color="#e67e22",
                                        lw=2, alpha=0.7),
                        zorder=5,
                    )

        # ============================================================
        # Phase 4: Animate shrinking (frames 65-104)
        # ============================================================
        elif frame <= 104:
            t = (frame - 65) / 39.0  # 0 → 1
            e = 1.0 - t * (1.0 - CCEI_STAR)
            # check violation at current e
            cost_0_of_1 = prices[0] @ quants[1]
            cost_1_of_0 = prices[1] @ quants[0]
            viol_0 = (e * expends[0]) >= cost_0_of_1 - 1e-9
            viol_1 = (e * expends[1]) >= cost_1_of_0 - 1e-9
            has_cycle = viol_0 and viol_1
            comp = ">=" if viol_0 else "<"
            status = "persists" if has_cycle else "broken"
            phase_txt.set_text("Step 4:  Binary Search over  $e$")
            desc_txt.set_text(
                f"Testing  $e$ = {e:.3f}.   "
                "Check GARP on deflated data.\n"
                f"$e \\cdot E_{{1,1}}$ = {e * expends[0]:.1f}  {comp}  "
                f"$E_{{1,2}}$ = {cost_0_of_1:.0f}?   Cycle {status}."
            )

            # ghost originals
            _draw_budget(ax, 0, 1.0, alpha_line=0.15, alpha_fill=0.02, lw=1)
            _draw_budget(ax, 1, 1.0, alpha_line=0.15, alpha_fill=0.02, lw=1)
            # current shrunken
            _draw_budget(ax, 0, e)
            _draw_budget(ax, 1, e)
            _draw_dot(ax, 0)
            _draw_dot(ax, 1)

            if has_cycle:
                _draw_pref_arrow(ax, 0, 1, PALETTE["highlight"], lw=2.5,
                                 alpha=0.6)
                _draw_pref_arrow(ax, 1, 0, PALETTE["highlight"], lw=2.5,
                                 alpha=0.6)
                ax.text(7.5, 8.0, "\u2716 GARP fails", fontsize=12,
                        fontweight="bold", color=PALETTE["highlight"],
                        ha="center", va="center", zorder=10)
            else:
                ax.text(7.5, 8.0, "\u2714 GARP holds", fontsize=12,
                        fontweight="bold", color=PALETTE["accent"],
                        ha="center", va="center", zorder=10)

            _draw_meter(e)

        # ============================================================
        # Phase 5: Critical threshold (frames 105-119)
        # ============================================================
        elif frame <= 119:
            e = CCEI_STAR
            phase_txt.set_text("Step 5:  GARP Satisfied at  $e^*$")
            desc_txt.set_text(
                f"At $e^*$ = {e:.3f}:  $e \\cdot E_{{1,1}}$ < $E_{{1,2}}$.  "
                f"$x_2$ no longer affordable.\n"
                f"The $R_0$ cycle breaks.  Deflated data is GARP-consistent."
            )

            _draw_budget(ax, 0, 1.0, alpha_line=0.15, alpha_fill=0.02, lw=1)
            _draw_budget(ax, 1, 1.0, alpha_line=0.15, alpha_fill=0.02, lw=1)
            _draw_budget(ax, 0, e)
            _draw_budget(ax, 1, e)
            _draw_dot(ax, 0)
            _draw_dot(ax, 1)

            ax.text(7.5, 8.0, "\u2714 GARP holds", fontsize=12,
                    fontweight="bold", color=PALETTE["accent"],
                    ha="center", va="center", zorder=10)

            _draw_meter(e, is_done=False)

            # highlight the gap where B is now outside
            if frame >= 110:
                alpha_note = min((frame - 110) / 5, 1.0)
                ax.text(
                    4.5, 1.0,
                    "$x_2$ now outside\ndeflated budget\n$e \\cdot E_{1,1}$",
                    fontsize=10, ha="center", va="center",
                    color=PALETTE["accent"], alpha=alpha_note,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor=PALETTE["accent"], lw=1.5,
                              alpha=alpha_note * 0.9),
                    zorder=10,
                )

        # ============================================================
        # Phase 6: Score reveal (frames 120-129)
        # ============================================================
        else:
            e = CCEI_STAR
            phase_txt.set_text("Result:  CCEI  =  sup$\\{e : \\text{GARP holds}\\}$")
            waste = (1.0 - e) * 100
            desc_txt.set_text(
                f"Critical ratio  $E_{{1,2}} / E_{{1,1}}$ = {E_matrix[0,1]/expends[0]:.3f}.  "
                f"Budget waste = {waste:.1f}%.\n"
                f"CCEI = 1.0 means perfectly rational (GARP satisfied)."
            )

            _draw_budget(ax, 0, 1.0, alpha_line=0.15, alpha_fill=0.02, lw=1)
            _draw_budget(ax, 1, 1.0, alpha_line=0.15, alpha_fill=0.02, lw=1)
            _draw_budget(ax, 0, e)
            _draw_budget(ax, 1, e)
            _draw_dot(ax, 0)
            _draw_dot(ax, 1)

            _draw_meter(e, is_done=True)

    # Text-change frames: phase starts + sub-phase text transitions
    text_frames = {0, 25, 37, 43, 50, 65, 105, 120}

    print("  Generating ccei_algorithm.gif...")
    _save_gif_variable_speed(fig, update, TOTAL_FRAMES, text_frames,
                              OUTPUT_DIR / "ccei_algorithm.gif")
    plt.close(fig)

# ---------------------------------------------------------------------------
# GIF: HM (Houtman-Maks Index) Pedagogical
# ---------------------------------------------------------------------------
def generate_hm_algorithm():
    """HM educational GIF — slow, well-spaced, no text overlap."""
    fig, ax = plt.subplots(figsize=(8, 9), facecolor=PALETTE["bg"])
    plt.subplots_adjust(top=0.78, bottom=0.14, left=0.08, right=0.95)

    TOTAL_FRAMES = 140
    INTERVAL = 3200  # ms — slow cadence

    # --- graph data: 5 nodes, 2 cycles sharing hub node 1 ---
    node_pos = {
        1: (0.0, 1.8),      # hub — center-ish
        2: (2.8, 3.5),      # top-right
        3: (2.8, 0.1),      # bottom-right
        4: (-2.8, 3.5),     # top-left
        5: (-2.8, 0.1),     # bottom-left
    }
    node_labels = {1: "$t$=1", 2: "$t$=2", 3: "$t$=3", 4: "$t$=4", 5: "$t$=5"}
    # cycle A: 1→2→3→1   cycle B: 1→4→5→1
    all_edges = [(1, 2), (2, 3), (3, 1), (1, 4), (4, 5), (5, 1)]
    cycle_a = {(1, 2), (2, 3), (3, 1)}
    cycle_b = {(1, 4), (4, 5), (5, 1)}
    NODE_R = 0.42

    # --- fixed title ---
    fig.text(
        0.04, 0.96,
        "Houtman-Maks  (HM)  Index",
        fontsize=15, fontweight="bold", color=PALETTE["accent"],
        family="monospace",
    )
    phase_txt = fig.text(0.04, 0.90, "", fontsize=13, fontweight="bold", color="#333")
    desc_txt = fig.text(0.04, 0.845, "", fontsize=11, style="italic", color="#555",
                        linespacing=1.5)
    score_txt = fig.text(0.50, 0.055, "", fontsize=16, fontweight="bold",
                         ha="center", va="center", color=PALETTE["accent"])
    counter_txt = fig.text(0.50, 0.02, "", fontsize=10, ha="center", va="center",
                           color="#666", family="monospace")

    def _draw_node(ax, nid, color, alpha=1.0, show_label=True):
        x, y = node_pos[nid]
        circle = plt.Circle((x, y), NODE_R, color=color, alpha=alpha, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, str(nid), fontsize=15, fontweight="bold", color="white",
                ha="center", va="center", zorder=11, alpha=alpha)
        if show_label:
            ax.text(x, y - NODE_R - 0.35, node_labels[nid], fontsize=9,
                    ha="center", va="top", color="#555", alpha=alpha, zorder=11)

    def _draw_edge(ax, u, v, color, lw=2.0, alpha=1.0):
        x1, y1 = node_pos[u]
        x2, y2 = node_pos[v]
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length < 0.01:
            return
        shrink = NODE_R + 0.08
        sx = x1 + shrink * dx / length
        sy = y1 + shrink * dy / length
        ex = x2 - shrink * dx / length
        ey = y2 - shrink * dy / length
        ax.annotate(
            "", xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                connectionstyle="arc3,rad=0.18", shrinkA=0, shrinkB=0,
            ),
            zorder=5,
        )

    def _draw_glow(ax, nid, color, alpha=0.3):
        """Draw a glowing ring around a node."""
        x, y = node_pos[nid]
        glow = plt.Circle((x, y), NODE_R + 0.15, color=color, alpha=alpha,
                           fill=False, lw=4, zorder=9)
        ax.add_patch(glow)

    def _setup_axes(ax):
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-1.2, 4.8)
        ax.axis("off")

    def update(frame):
        ax.clear()
        _setup_axes(ax)

        # ============================================================
        # Phase 1: Meet the trips (frames 0-24)
        # ============================================================
        if frame <= 24:
            phase_txt.set_text("Step 1:  Five Observations  ($T$ = 5)")
            desc_txt.set_text(
                "A BehaviorLog with 5 observations ($p_t$, $x_t$).\n"
                "Each observation is a node in the $R_0$ preference graph."
            )
            score_txt.set_text("")
            counter_txt.set_text("")

            # nodes appear one at a time: 1 at f=2, 2 at f=6, 3 at f=10, 4 at f=14, 5 at f=18
            for i, nid in enumerate([1, 2, 3, 4, 5]):
                appear = 2 + i * 5
                if frame >= appear:
                    alpha = min((frame - appear) / 3, 1.0)
                    _draw_node(ax, nid, PALETTE["node"], alpha=alpha)

        # ============================================================
        # Phase 2: Build preference edges (frames 25-54)
        # ============================================================
        elif frame <= 54:
            phase_txt.set_text("Step 2:  Build $R_0$ Graph")
            desc_txt.set_text(
                "Edge $i \\to j$ when $E_{ii} \\geq E_{ij}$  ($x_j$ was affordable at $t$=$i$).\n"
                "$x_i$ chosen over $x_j$  $\\Rightarrow$  $x_i \\; R_0 \\; x_j$  (revealed preferred)."
            )
            score_txt.set_text("")
            counter_txt.set_text("")

            # all nodes visible
            for nid in node_pos:
                _draw_node(ax, nid, PALETTE["node"])

            # edges appear one at a time: 5 frames per edge
            for i, (u, v) in enumerate(all_edges):
                appear = 25 + i * 5
                if frame >= appear:
                    alpha = min((frame - appear) / 3, 1.0)
                    _draw_edge(ax, u, v, PALETTE["edge"], lw=2.0, alpha=alpha)

        # ============================================================
        # Phase 3: Spot the cycles (frames 55-79)
        # ============================================================
        elif frame <= 79:
            score_txt.set_text("")

            # all nodes + edges present
            for nid in node_pos:
                _draw_node(ax, nid, PALETTE["node"])

            if frame <= 66:
                phase_txt.set_text("Step 3:  Find SCCs  (Strongly Connected Components)")
                desc_txt.set_text(
                    "SCC {1,2,3}:  $1 \\to 2 \\to 3 \\to 1$.  A cycle in $R_0$.\n"
                    "Cycles in the preference graph $\\Rightarrow$ GARP violations."
                )
                counter_txt.set_text("SCCs with cycles: 1")
                # draw all edges, highlight cycle A
                for u, v in all_edges:
                    if (u, v) in cycle_a:
                        pulse = 0.6 + 0.4 * abs(np.sin(frame * 0.3))
                        _draw_edge(ax, u, v, PALETTE["highlight"], lw=3.5,
                                   alpha=pulse)
                    else:
                        _draw_edge(ax, u, v, PALETTE["edge"], lw=1.5, alpha=0.4)
                # glow cycle A nodes
                for nid in [1, 2, 3]:
                    _draw_glow(ax, nid, PALETTE["highlight"], alpha=0.25)
            else:
                phase_txt.set_text("Step 3:  Two SCCs with Cycles")
                desc_txt.set_text(
                    "SCC {1,4,5}:  $1 \\to 4 \\to 5 \\to 1$.  Second cycle.\n"
                    "Obs 1 is in both SCCs $\\Rightarrow$ high degree in $R_0$."
                )
                counter_txt.set_text("SCCs with cycles: 2  |  Obs 1 in both")
                # highlight both cycles
                for u, v in all_edges:
                    if (u, v) in cycle_a or (u, v) in cycle_b:
                        pulse = 0.6 + 0.4 * abs(np.sin(frame * 0.3))
                        _draw_edge(ax, u, v, PALETTE["highlight"], lw=3.5,
                                   alpha=pulse)
                    else:
                        _draw_edge(ax, u, v, PALETTE["edge"], lw=1.5, alpha=0.4)
                for nid in [1, 2, 3, 4, 5]:
                    _draw_glow(ax, nid, PALETTE["highlight"], alpha=0.2)

        # ============================================================
        # Phase 4: Try removing node 3 (frames 80-99)
        # ============================================================
        elif frame <= 99:
            score_txt.set_text("")

            if frame <= 89:
                phase_txt.set_text("Step 4:  Greedy FVS \u2014 Try Removing Obs 3")
                desc_txt.set_text(
                    "Drop obs 3 from the Feedback Vertex Set.  SCC {1,2,3} breaks  \u2714\n"
                    "But SCC {1,4,5} still has a cycle...  \u2716"
                )
                counter_txt.set_text("FVS = {3}  |  Remaining cycles: 1")

                active = {1, 2, 4, 5}
                for nid in node_pos:
                    if nid in active:
                        _draw_node(ax, nid, PALETTE["node"])
                    else:
                        _draw_node(ax, nid, PALETTE["grid"], alpha=0.2)

                for u, v in all_edges:
                    if u not in active or v not in active:
                        continue
                    if (u, v) in cycle_b:
                        pulse = 0.6 + 0.4 * abs(np.sin(frame * 0.3))
                        _draw_edge(ax, u, v, PALETTE["highlight"], lw=3, alpha=pulse)
                    else:
                        _draw_edge(ax, u, v, PALETTE["edge"], lw=1.5, alpha=0.5)

                # show checkmark for cycle A, X for cycle B
                ax.text(3.5, 2.0, "SCC {1,2,3}\n\u2714 Broken", fontsize=10,
                        ha="center", color=PALETTE["accent"], fontweight="bold")
                ax.text(-3.5, 2.0, "SCC {1,4,5}\n\u2716 Cycle", fontsize=10,
                        ha="center", color=PALETTE["highlight"], fontweight="bold")

            else:
                # restore node 3, show all again
                phase_txt.set_text("Step 4:  Greedy FVS \u2014 Pick Highest Degree")
                desc_txt.set_text(
                    "Removing obs 3 only breaks one SCC.\n"
                    "Greedy heuristic: remove the node with highest degree."
                )
                counter_txt.set_text("Obs 1: degree=4 (highest)  |  Re-evaluating...")

                for nid in node_pos:
                    _draw_node(ax, nid, PALETTE["node"])
                for u, v in all_edges:
                    _draw_edge(ax, u, v, PALETTE["highlight"], lw=2, alpha=0.5)

                # highlight node 1 as special
                _draw_glow(ax, 1, "#e67e22", alpha=0.5)

        # ============================================================
        # Phase 5: Remove the hub — node 1 (frames 100-124)
        # ============================================================
        elif frame <= 124:
            if frame <= 107:
                phase_txt.set_text("Step 5:  FVS = {Obs 1}  (Highest Degree)")
                desc_txt.set_text(
                    "Obs 1 participates in both SCCs (degree=4 in $R_0$).\n"
                    "Add obs 1 to the Feedback Vertex Set and remove it."
                )
                counter_txt.set_text("FVS = {1}  |  Removing from R\u2080 graph...")

                for nid in node_pos:
                    _draw_node(ax, nid, PALETTE["node"])
                for u, v in all_edges:
                    _draw_edge(ax, u, v, PALETTE["edge"], lw=1.5, alpha=0.5)
                _draw_glow(ax, 1, "#e67e22", alpha=0.6)

                # degree annotation
                x1, y1 = node_pos[1]
                ax.text(x1 + 0.9, y1 + 0.5, "deg=4\n(max)", fontsize=9,
                        color="#e67e22", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="#e67e22", lw=1.2, alpha=0.85))

            else:
                phase_txt.set_text("Step 5:  $R_0$ Graph Is Now a DAG")
                desc_txt.set_text(
                    "Without obs 1, all SCCs are trivial (size 1).\n"
                    "Remaining observations are GARP-consistent."
                )
                counter_txt.set_text("FVS = {1}  |  |FVS| = 1  |  All SCCs trivial")

                fade = max(0.15, 1.0 - (frame - 108) / 6)
                active = {2, 3, 4, 5}
                for nid in node_pos:
                    if nid in active:
                        _draw_node(ax, nid, PALETTE["accent"])
                    else:
                        _draw_node(ax, nid, PALETTE["grid"], alpha=min(fade, 0.2),
                                   show_label=False)

                for u, v in all_edges:
                    if u not in active or v not in active:
                        continue
                    _draw_edge(ax, u, v, PALETTE["accent"], lw=2, alpha=0.6)

                # show both cycles broken
                ax.text(3.5, 2.0, "SCC {2,3}\ntrivial \u2714", fontsize=10, ha="center",
                        color=PALETTE["accent"], fontweight="bold")
                ax.text(-3.5, 2.0, "SCC {4,5}\ntrivial \u2714", fontsize=10, ha="center",
                        color=PALETTE["accent"], fontweight="bold")

        # ============================================================
        # Phase 6: Score reveal (frames 125-139)
        # ============================================================
        else:
            phase_txt.set_text("Result:  HM  =  (T \u2212 |FVS|) / T")
            desc_txt.set_text(
                "Max GARP-consistent subset: 4 of 5 observations.\n"
                "Greedy FVS found |FVS|=1.  Score = (5\u22121)/5 = 0.80."
            )

            active = {2, 3, 4, 5}
            for nid in node_pos:
                if nid in active:
                    _draw_node(ax, nid, PALETTE["accent"])
                else:
                    _draw_node(ax, nid, PALETTE["grid"], alpha=0.15,
                               show_label=False)
            for u, v in all_edges:
                if u not in active or v not in active:
                    continue
                _draw_edge(ax, u, v, PALETTE["accent"], lw=2, alpha=0.6)

            alpha_score = min((frame - 125) / 5, 1.0)
            score_txt.set_text("HM = (T \u2212 |FVS|) / T = 4/5 = 0.80")
            score_txt.set_alpha(alpha_score)
            counter_txt.set_text("80% of observations satisfy GARP")

    # Text-change frames: phase starts + sub-phase text transitions
    text_frames = {0, 25, 55, 67, 80, 90, 100, 108, 125}

    print("  Generating hm_algorithm.gif...")
    _save_gif_variable_speed(fig, update, TOTAL_FRAMES, text_frames,
                              OUTPUT_DIR / "hm_algorithm.gif")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Generating documentation GIFs...")

    generate_budget_hero()
    generate_menu_hero()
    generate_floyd_warshall()
    generate_scc_tarjan()
    generate_power_analysis()
    generate_engine_throughput()
    generate_attention_decay()
    generate_ccei_algorithm()
    generate_hm_algorithm()

    print(f"\nAll GIFs saved to {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*.gif")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
