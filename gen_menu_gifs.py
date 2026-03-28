import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import io

plt.switch_backend("Agg")

out_dir = Path("docs/_static")
out_dir.mkdir(parents=True, exist_ok=True)
DPI = 100

PALETTE = {
    "bg": "#fafafa", "edge": "#4a4a4a", "node": "#5b8def",
    "highlight": "#e74c3c", "secondary": "#95a5a6", "accent": "#27ae60"
}

def draw_box(ax, x, y, text, w, h, bg, txt_c, lw=1, ec="#95a5a6"):
    rect = plt.Rectangle((x-w/2, y-h/2), w, h, facecolor=bg, edgecolor=ec, lw=lw, zorder=5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", color=txt_c, fontweight="bold", zorder=6)

# 1. Deterministic
def gen_deterministic():
    fig, ax = plt.subplots(figsize=(4, 3), facecolor=PALETTE["bg"])
    items = ["Laptop", "Tablet", "Phone"]
    frames = 20
    def update(f):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.5, 1.5)
        ax.axis("off")
        ax.set_title("Deterministic\nMenuChoiceLog", fontsize=12, fontweight="bold", pad=10, color="#333")
        
        for i, item in enumerate(items):
            y = 0.6 - i*0.6
            if f > 8 and i == 0:
                draw_box(ax, 0, y, item, 2.0, 0.4, PALETTE["node"], "white", lw=2, ec=PALETTE["node"])
                ax.text(1.2, y, "✓", color=PALETTE["node"], fontsize=16, va="center")
            else:
                draw_box(ax, 0, y, item, 2.0, 0.4, "white", PALETTE["secondary"])
    
    anim = FuncAnimation(fig, update, frames=frames, interval=250)
    anim.save(out_dir / "deterministic.gif", writer="pillow", dpi=DPI)
    plt.close(fig)

# 2. Stochastic
def gen_stochastic():
    fig, ax = plt.subplots(figsize=(4, 3), facecolor=PALETTE["bg"])
    items = [("Laptop", 0.6), ("Tablet", 0.3), ("Phone", 0.1)]
    frames = 20
    def update(f):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.5, 1.5)
        ax.axis("off")
        ax.set_title("Stochastic\nStochasticChoiceLog", fontsize=12, fontweight="bold", pad=10, color="#333")
        
        prog = min(1.0, f / 12.0)
        for i, (item, prob) in enumerate(items):
            y = 0.6 - i*0.6
            w = 2.0
            h = 0.4
            # Draw base box
            draw_box(ax, 0, y, f"{item} ({int(prob*100*prog)}%)", w, h, "white", PALETTE["edge"])
            # Draw fill
            if prog > 0:
                fill_w = w * prob * prog
                fill_x = -w/2 + fill_w/2
                rect = plt.Rectangle((-w/2, y-h/2), fill_w, h, facecolor=PALETTE["accent"], alpha=0.3, edgecolor="none", zorder=4)
                ax.add_patch(rect)
    
    anim = FuncAnimation(fig, update, frames=frames, interval=250)
    anim.save(out_dir / "stochastic.gif", writer="pillow", dpi=DPI)
    plt.close(fig)

# 3. Risk
def gen_risk():
    fig, ax = plt.subplots(figsize=(4, 3), facecolor=PALETTE["bg"])
    frames = 20
    def update(f):
        ax.clear()
        ax.set_facecolor(PALETTE["bg"])
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.5, 1.5)
        ax.axis("off")
        ax.set_title("Risk / Lotteries\nRiskChoiceLog", fontsize=12, fontweight="bold", pad=10, color="#333")
        
        # Risk A: 50% \$100 / 50% \$0
        draw_box(ax, 0, 0.4, "Gamble A\n50% \$100 | 50% \$0", 2.2, 0.6, "white", PALETTE["secondary"])
        
        # Risk B: 100% \$40
        draw_box(ax, 0, -0.6, "Gamble B\n100% \$40", 2.2, 0.6, "white", PALETTE["secondary"])
        
        if f > 8:
            draw_box(ax, 0, -0.6, "Gamble B\n100% \$40", 2.2, 0.6, PALETTE["highlight"], "white", lw=2, ec=PALETTE["highlight"])
            ax.text(1.3, -0.6, "✓", color=PALETTE["highlight"], fontsize=16, va="center")
            
    anim = FuncAnimation(fig, update, frames=frames, interval=250)
    anim.save(out_dir / "risk.gif", writer="pillow", dpi=DPI)
    plt.close(fig)

if __name__ == '__main__':
    print("Generating deterministic...")
    gen_deterministic()
    print("Generating stochastic...")
    gen_stochastic()
    print("Generating risk...")
    gen_risk()
    print("Done")
