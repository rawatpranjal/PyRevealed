#!/usr/bin/env python3
"""Generate animated GIF visualizations for RTD documentation."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from pathlib import Path

plt.switch_backend("Agg")

OUTPUT_DIR = Path(__file__).parent / "_static"
DPI = 100
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
# ---------------------------------------------------------------------------
def generate_budget_hero():
    scenarios = [
        {
            "title": "1. Consistent Choices (CCEI = 1.0)",
            "desc": "Perfectly rational behavior. No cycles, clean preferences.",
            "prices": np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0]]),
            "quants": np.array([[4.0, 4.0], [5.0, 1.0], [1.0, 5.0]]),
            "accent": PALETTE["accent"]
        },
        {
            "title": "2. Mild Inconsistency (CCEI = 0.88)",
            "desc": "Minor intersection: $x^1$ chosen over $x^2$ & vice-versa.",
            "prices": np.array([[1.0, 2.0], [2.0, 1.0]]),
            "quants": np.array([[2.0, 3.0], [3.0, 2.0]]),
            "accent": "#e67e22"
        },
        {
            "title": "3. Highly Inconsistent (CCEI = 0.54)",
            "desc": "Severe preference cycles inside budgets. Very irrational.",
            "prices": np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [1.2, 1.2]]),
            "quants": np.array([[2.0, 2.5], [2.5, 2.0], [2.0, 2.0], [4.0, 4.0]]),
            "accent": PALETTE["highlight"]
        }
    ]

    TOTAL_FRAMES = 85
    fig, axes = plt.subplots(3, 1, figsize=(7, 11.5), facecolor=PALETTE["bg"])
    plt.subplots_adjust(hspace=0.65, top=0.92, bottom=0.06, left=0.1, right=0.95)

    def _draw_arrow(ax, x0, y0, x1, y1, color, lw=1.5):
        dx, dy = x1 - x0, y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        if length < 0.01: return
        shrink = 0.35
        sx, sy = x0 + shrink * dx / length, y0 + shrink * dy / length
        ex, ey = x1 - shrink * dx / length, y1 - shrink * dy / length
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0))

    def update(frame):
        for idx, (ax, sc) in enumerate(zip(axes, scenarios)):
            ax.clear()
            ax.set_facecolor(PALETTE["bg"])
            ax.set_title(sc["title"], fontsize=13, fontweight="bold", pad=26, loc="left", color=sc["accent"])
            
            prices, quants = sc["prices"], sc["quants"]
            expenditures = np.sum(prices * quants, axis=1)
            n_total = len(prices)
            obs_colors = ["#5b8def", "#8e44ad", "#27ae60", "#e74c3c"]
            
            ax.set_xlim(-0.2, 8.5)
            ax.set_ylim(-0.2, 8.5)
            if idx == 2:
                ax.set_xlabel("Good 1", fontsize=11)
            ax.set_ylabel("Good 2", fontsize=11)
            ax.grid(True, alpha=0.15, color=PALETTE["grid"])
            ax.text(0.00, 1.02, sc["desc"], transform=ax.transAxes, fontsize=10, style="italic")

            budget_prefs = []
            for i in range(n_total):
                for j in range(n_total):
                    if i != j and prices[i] @ quants[i] >= (prices[i] @ quants[j]) - 1e-5:
                        budget_prefs.append((i, j))

            n_obs = min((frame // 8) + 1, n_total) if frame < 35 else n_total

            for t in range(n_obs):
                p0, p1 = prices[t]
                budget = expenditures[t]
                if p0 > 0 and p1 > 0:
                    ax.plot([0, budget/p0], [budget/p1, 0], color=obs_colors[t], lw=2.5, alpha=0.5)
                    ax.fill_between([0, budget/p0], [budget/p1, 0], alpha=0.06, color=obs_colors[t])
                ax.scatter(quants[t, 0], quants[t, 1], color=obs_colors[t], s=120, zorder=5, edgecolors="white", lw=2)
                ax.annotate(f"$x^{t+1}$", (quants[t, 0] + 0.2, quants[t, 1] + 0.2), 
                            fontsize=11, fontweight="bold", color=obs_colors[t])

            if frame >= 35:
                n_arrows = min((frame - 35) // 3 + 1, len(budget_prefs)) if frame < 65 else len(budget_prefs)
                for i_arr in range(n_arrows):
                    i, j = budget_prefs[i_arr]
                    _draw_arrow(ax, quants[i, 0], quants[i, 1],
                                quants[j, 0], quants[j, 1],
                                color=PALETTE["edge"] if "Consistent" in sc["title"] else PALETTE["highlight"], lw=2.0)

            if frame >= 65:
                alpha = min((frame - 65) / 5, 1.0)
                if "Consistent" in sc["title"]:
                    box_text = "Consistent"
                elif "Mild" in sc["title"]:
                    box_text = "2-Cycle Violation"
                else:
                    box_text = "Severe Violations"
                ax.text(6.5, 7.0, box_text, fontsize=12, fontweight="bold", ha="center",
                        color=sc["accent"], alpha=alpha,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=sc["accent"], alpha=alpha))

    print("  Generating budget_hero.gif...")
    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=400)
    anim.save(OUTPUT_DIR / "budget_hero.gif", writer="pillow", dpi=DPI)
    plt.close(fig)

# ---------------------------------------------------------------------------
# GIF 1B: Landing Page — Menu Choice
# ---------------------------------------------------------------------------
def generate_menu_hero():
    scenarios = [
        {
            "title": "1. Consistent Choices (HM = 1.00)",
            "desc": "Satisfies SARP. 'A' is robustly preferred to B and C across contexts.",
            "menus": [({"A", "B", "C"}, "A"), ({"B", "C"}, "B"), ({"A", "C"}, "A")],
            "accent": PALETTE["accent"]
        },
        {
            "title": "2. Mild Inconsistency (HM = 0.66)",
            "desc": "Adding 'C' changes preference from A to B. Direct WARP violation.",
            "menus": [({"A", "B"}, "A"), ({"A", "B", "C"}, "B"), ({"B", "C"}, "C")],
            "accent": "#e67e22"
        },
        {
            "title": "3. Highly Inconsistent (HM = 0.50)",
            "desc": "Severe intransitive cycles: A ≻ B ≻ C ≻ D ≻ A.",
            "menus": [({"A", "B"}, "A"), ({"B", "C"}, "B"), ({"C", "D"}, "C"), ({"A", "D"}, "D")],
            "accent": PALETTE["highlight"]
        }
    ]
    
    TOTAL_FRAMES = 85
    fig, axes = plt.subplots(3, 1, figsize=(7, 11.5), facecolor=PALETTE["bg"])
    plt.subplots_adjust(hspace=0.65, top=0.92, bottom=0.06, left=0.1, right=0.95)
    
    all_items = ["A", "B", "C", "D"]
    item_colors = {"A": "#5b8def", "B": "#e67e22", "C": "#27ae60", "D": "#8e44ad"}

    def update(frame):
        for idx, (ax, sc) in enumerate(zip(axes, scenarios)):
            ax.clear()
            ax.set_facecolor(PALETTE["bg"])
            ax.set_title(sc["title"], fontsize=13, fontweight="bold", pad=26, loc="left", color=sc["accent"])
            ax.text(0.00, 1.02, sc["desc"], transform=ax.transAxes, fontsize=10, style="italic")
            
            menus = sc["menus"]
            n_total = len(menus)
            
            ax.set_xlim(-3, 3)
            ax.set_ylim(-len(menus)*1.3 - 0.5, 0.5)
            ax.set_aspect("equal")
            ax.axis("off")

            menu_prefs = []
            for m_set, chosen in menus:
                for item in m_set:
                    if item != chosen:
                        menu_prefs.append((chosen, item))
            
            n_obs = min((frame // 8) + 1, n_total) if frame < 35 else n_total

            for m_idx in range(n_obs):
                menu_set, chosen = menus[m_idx]
                y_base = -1.2 * m_idx - 0.8
                sorted_items = sorted(list(menu_set))
                n_items = len(sorted_items)
                x_positions = np.linspace(-1.0 * (n_items-1)/2, 1.0 * (n_items-1)/2, n_items) if n_items > 1 else [0]
                
                ax.plot([min(x_positions)-0.5, max(x_positions)+0.5], [y_base-0.45, y_base-0.45], 
                        color=PALETTE["secondary"], lw=1.5, alpha=0.4)
                ax.text(-2.5, y_base, f"Menu {m_idx+1}:", fontsize=11, va="center", color=PALETTE["secondary"])

                for i_idx, item in enumerate(sorted_items):
                    x = x_positions[i_idx]
                    is_chosen = (item == chosen)
                    radius = 0.28
                    color = item_colors[item]
                    alpha = 1.0 if is_chosen else 0.3
                    lw = 2.5 if is_chosen else 1.0
                    circle = plt.Circle((x, y_base), radius, facecolor=color, alpha=alpha, 
                                        edgecolor="white" if is_chosen else PALETTE["secondary"], lw=lw, zorder=10)
                    ax.add_patch(circle)
                    ax.text(x, y_base, item, ha="center", va="center", fontsize=13,
                            fontweight="bold" if is_chosen else "normal",
                            color="white" if is_chosen else PALETTE["edge"], zorder=11)

            if frame >= 35:
                n_pairs = min((frame - 35) // 3 + 1, len(menu_prefs)) if frame < 65 else len(menu_prefs)
                if n_pairs > 0:
                    y_text_start = -0.5
                    ax.text(1.8, y_text_start, "Inferred:", fontsize=11, fontweight="bold", color=PALETTE["secondary"])
                    for p_idx in range(n_pairs):
                        ch, unch = menu_prefs[p_idx]
                        ax.text(1.8, y_text_start - 0.4 - 0.35*p_idx, f"{ch} $\succ$ {unch}", 
                                fontsize=12, color=PALETTE["edge"])

            if frame >= 65:
                alpha = min((frame - 65) / 5, 1.0)
                if "Consistent" in sc["title"]:
                    box_text = "SARP Consistent"
                elif "Mild" in sc["title"]:
                    box_text = "WARP Violated"
                else:
                    box_text = "SARP Violated"
                
                ax.text(1.8, -len(menus)*1.3 + 0.3, box_text, fontsize=11, fontweight="bold", ha="center",
                        color=sc["accent"], alpha=alpha,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=sc["accent"], alpha=alpha))

    print("  Generating menu_hero.gif...")
    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=400)
    anim.save(OUTPUT_DIR / "menu_hero.gif", writer="pillow", dpi=DPI)
    plt.close(fig)

def generate_floyd_warshall():
    """Preference graph filling in through transitive closure."""
    n = 5
    labels = [f"$x^{i+1}$" for i in range(n)]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    pos = np.column_stack([np.cos(angles), np.sin(angles)]) * 1.8

    # Direct edges: 0->1, 1->2, 2->3, 3->4, 0->3
    direct = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)]
    adj = np.zeros((n, n), dtype=bool)
    for i, j in direct:
        adj[i, j] = True

    states = [adj.copy()]
    focus_nodes = [] # specifically (i, k, j) being tested
    
    for k in range(n):
        step_focus = []
        for i in range(n):
            for j in range(n):
                if adj[i, k] and adj[k, j] and not adj[i, j]:
                    adj[i, j] = True
                    step_focus.append((i, k, j))
        states.append(adj.copy())
        focus_nodes.append(step_focus)

    fig, ax = plt.subplots(figsize=(5.5, 6.5), facecolor=PALETTE["bg"])

    def draw_arrow(ax, start, end, color, lw=1.5, style="-", alpha=1.0):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        if length < 0.01: return
        shrink = 0.3
        sx, sy = start[0] + shrink * dx / length, start[1] + shrink * dy / length
        ex, ey = end[0] - shrink * dx / length, end[1] - shrink * dy / length
        ax.annotate(
            "", xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=style, shrinkA=0, shrinkB=0, alpha=alpha),
        )

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-3.8, 2.8)
        ax.set_aspect("equal")
        ax.axis("off")

        k_label = ""
        desc = ""
        
        if frame < 6:
            current = states[0]
            k_label = "Phase 1: Direct Preferences ($R_0$)"
            desc = "Building basic connections between choices"
            active_triads = []
        elif frame < 6 + n*4:
            k = (frame - 6) // 4
            step_frame = (frame - 6) % 4
            current = states[k] if step_frame < 2 else states[k+1]
            active_triads = focus_nodes[k] if step_frame in [1, 2] else []
            
            k_label = f"Phase 2: Triplet Search (via $x^{k+1}$) — O(T³)"
            if step_frame == 0: desc = f"Scanning paths through $x^{k+1}$..."
            elif step_frame == 1: desc = f"Found path! $x_i \to x^{k+1} \to x_j$"
            elif step_frame == 2: desc = "Adding indirect preference bridge."
            else: desc = "Complete."
        else:
            current = states[-1]
            k_label = "Complete Transitive Closure ($R^*$)"
            desc = "All indirect preferences inferred (Costly: O(T³))"
            active_triads = []

        direct_set = set(direct)
        
        active_edges_highlight = []
        inferred_new = []
        for triad in active_triads:
            i, k_node, j = triad
            active_edges_highlight.append((i, k_node))
            active_edges_highlight.append((k_node, j))
            inferred_new.append((i, j))

        # draw edges
        for i in range(n):
            for j in range(n):
                if i != j and current[i, j]:
                    is_direct = (i, j) in direct_set
                    is_highlighted = (i, j) in active_edges_highlight
                    is_new = (i, j) in inferred_new
                    
                    if is_new:
                        color = PALETTE["highlight"]
                        lw = 2.5
                        style = "--"
                    elif is_highlighted:
                        color = PALETTE["python"]
                        lw = 3.0
                        style = "-" if is_direct else "--"
                    else:
                        color = PALETTE["edge"] if is_direct else PALETTE["secondary"]
                        lw = 2.0 if is_direct else 1.5
                        style = "-" if is_direct else "--"
                    draw_arrow(ax, pos[i], pos[j], color, lw, style)

        # draw nodes
        for i in range(n):
            circle = plt.Circle(pos[i], 0.25, color=PALETTE["node"], zorder=10)
            ax.add_patch(circle)
            ax.text(pos[i][0], pos[i][1], labels[i], ha="center", va="center", fontsize=11,
                    color=PALETTE["node_text"], fontweight="bold", zorder=11)

        ax.set_title("1. Traditional: Floyd-Warshall", fontsize=13, fontweight="bold", color="#333", pad=26)
        ax.text(0, -2.8, k_label, ha="center", fontsize=11, fontweight="bold", color=PALETTE["python"] if "Search" in k_label else PALETTE["edge"])
        ax.text(0, -3.2, desc, ha="center", fontsize=10, style="italic", color=PALETTE["secondary"])

    total_frames = 6 + n*4 + 10
    anim = FuncAnimation(fig, update, frames=total_frames, interval=750)
    anim.save(OUTPUT_DIR / "floyd_warshall.gif", writer="pillow", dpi=DPI)
    plt.close(fig)
    print("  floyd_warshall.gif updated with pedagogical flow")


# ---------------------------------------------------------------------------
# GIF 3: Tarjan's SCC (O(T^2))
# ---------------------------------------------------------------------------
def generate_scc_tarjan():
    """Tarjan's SCC forming components and detecting strict cycle violation."""
    n = 6
    labels = ["$x^1$", "$x^2$", "$x^3$", "$x^4$", "$x^5$", "$x^6$"]
    # Group [0, 1, 2] is an SCC, [3, 4] is another, 5 is a sink
    pos = np.array([
        [-1.0, 1.2], [1.0, 1.2], [0.0, -0.5],  # SCC 1
        [-1.5, -1.8], [1.5, -1.8],             # SCC 2
        [0.0, -2.5]                            # Sink
    ])

    weak_edges = [(0, 1), (1, 2), (2, 0), (2, 3), (2, 4), (3, 4), (4, 3), (3, 5), (4, 5)]
    # Violation: (2, 0) is actually a strict edge P_0 inside SCC 1
    strict_edges = [(2, 0)]
    scc_1 = [0, 1, 2]

    fig, ax = plt.subplots(figsize=(5.5, 6.5), facecolor=PALETTE["bg"])

    def draw_edge(ax, i, j, color, lw=2.0, style="-"):
        dx, dy = pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]
        length = np.sqrt(dx**2 + dy**2)
        shrink = 0.32
        sx, sy = pos[i][0] + shrink * dx / length, pos[i][1] + shrink * dy / length
        ex, ey = pos[j][0] - shrink * dx / length, pos[j][1] - shrink * dy / length
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=style, shrinkA=0, shrinkB=0))

    def update(frame):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-4.2, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")

        scc_active = False
        violation_active = False

        if frame < 8:
            visible = int((frame / 8) * len(weak_edges))
            for idx in range(visible):
                draw_edge(ax, weak_edges[idx][0], weak_edges[idx][1], PALETTE["edge"], lw=1.5)
            subtitle = "Phase 1: Direct Preference Graph ($R_0$)"
            desc = "Cycles just imply mutual indifference (not violations)!"
        elif frame < 18:
            for u, v in weak_edges:
                draw_edge(ax, u, v, PALETTE["edge"], lw=1.5)
            subtitle = "Phase 2: Group Indifferences (O(T²))"
            desc = "Tarjan's bundles all cycles into SCC components."
            scc_active = True
        elif frame < 26:
            for u, v in weak_edges:
                if (u, v) == strict_edges[0]:
                    draw_edge(ax, u, v, PALETTE["highlight"], lw=3.0)
                else:
                    color = PALETTE["rust"] if u in scc_1 and v in scc_1 else PALETTE["edge"]
                    draw_edge(ax, u, v, color, lw=1.5)
            subtitle = "Phase 3: Screen SCCs for Strict Arcs ($P_0$)"
            desc = "If any arc inside the indifferent group is Strict, it's exploited!"
            scc_active = True
            violation_active = True
        else:
            for u, v in weak_edges:
                if (u, v) == strict_edges[0]:
                    draw_edge(ax, u, v, PALETTE["highlight"], lw=3.0)
                else:
                    color = PALETTE["rust"] if u in scc_1 and v in scc_1 else PALETTE["secondary"]
                    draw_edge(ax, u, v, color, lw=1.5)
            subtitle = "GARP Violation Flagged"
            desc = "Bypassed O(T³) transitive closure entirely (Theorem 1)."
            scc_active = True
            violation_active = True
            
            # draw large cross over the strict edge
            ax.text(pos[2][0]-0.6, pos[2][1] + 1.2, "✘", color=PALETTE["highlight"], fontsize=50, ha="center", va="center", alpha=0.9, zorder=0)

        if scc_active:
            # Highlight SCC1 bubble
            circle_scc = plt.Circle((0, 0.4), 1.6, fill=True, color=PALETTE["rust"], alpha=0.15, zorder=1)
            ax.add_patch(circle_scc)
            if frame > 12:
                # Highlight SCC2
                circle_scc2 = plt.Circle((0, -1.8), 2.2, fill=True, color=PALETTE["accent"], alpha=0.1, zorder=1)
                ax.add_patch(circle_scc2)
            
            if scc_active and not violation_active:
                ax.text(2.0, 1.2, "SCC: Mutual\nIndifference\n$x^1 \sim x^2 \sim x^3$", color=PALETTE["rust"], fontsize=10, fontweight="bold", alpha=min((frame-8)/3, 1.0))

        # draw nodes
        for i in range(n):
            node_color = PALETTE["node"]
            if scc_active:
                if i in scc_1: node_color = PALETTE["rust"]
                elif i in [3,4]: node_color = PALETTE["accent"]
            
            circle = plt.Circle(pos[i], 0.25, color=node_color, zorder=10)
            ax.add_patch(circle)
            ax.text(pos[i][0], pos[i][1], labels[i], ha="center", va="center", fontsize=11,
                    color=PALETTE["node_text"], fontweight="bold", zorder=11)

        ax.set_title("2. Modern Engine: Tarjan's SCC", fontsize=13, fontweight="bold", color="#333", pad=26)
        ax.text(0, -3.4, subtitle, ha="center", fontsize=11, fontweight="bold", color=PALETTE["highlight"] if "Flag" in subtitle else PALETTE["edge"])
        ax.text(0, -3.8, desc, ha="center", fontsize=10, style="italic", color=PALETTE["secondary"])

    anim = FuncAnimation(fig, update, frames=34, interval=750)
    anim.save(OUTPUT_DIR / "scc_tarjan.gif", writer="pillow", dpi=DPI)
    plt.close(fig)
    print("  scc_tarjan.gif updated with pedagogical flow")



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
