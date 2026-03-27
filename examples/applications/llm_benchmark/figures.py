#!/usr/bin/env python3
"""Stage 4: Generate benchmark visualization figures.

Reads summary.json and produces a 2x3 panel showing:
  (a) HM efficiency heatmap by scenario x prompt (split by model)
  (b) SARP pass/fail grid
  (c) Instinct vs reasoning scatter
  (d) Violation count comparison
  (e) Prompt type consistency ranking
  (f) Worst-case preference graph

Usage:
    python -m llm_benchmark.figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .config import ALL_SCENARIOS, MODEL_CONFIGS

RESULTS_DIR = Path(__file__).parent / "data" / "results"
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "docs" / "_static"

PROMPT_ORDER = ["minimal", "decision_tree", "conservative", "aggressive", "chain_of_thought"]
PROMPT_LABELS = ["Minimal", "Decision Tree", "Conservative", "Aggressive", "Chain-of-Thought"]
SCENARIO_ORDER = list(ALL_SCENARIOS.keys())
SCENARIO_LABELS = [ALL_SCENARIOS[s].display_name for s in SCENARIO_ORDER]
MODEL_SLUGS = {m["name"]: m["slug"] for m in MODEL_CONFIGS}


def load_summary() -> dict:
    path = RESULTS_DIR / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No summary at {path}. Run analyze.py first.")
    with open(path) as f:
        return json.load(f)


def _get_metric(summary: dict, scenario: str, prompt: str, model_name: str, metric: str, default=0.0):
    """Safely extract a metric from the summary."""
    slug = MODEL_SLUGS.get(model_name, model_name)
    key = f"{prompt}__{slug}"
    return summary.get(scenario, {}).get(key, {}).get(metric, default)


def generate_panel(summary: dict) -> None:
    """Generate the 2x3 summary panel."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("LLM Enterprise Consistency Benchmark", fontsize=16, fontweight="bold", y=0.98)

    model_names = [m["name"] for m in MODEL_CONFIGS]

    # (a) HM Efficiency Heatmap — side by side
    ax = axes[0, 0]
    n_scenarios = len(SCENARIO_ORDER)
    n_prompts = len(PROMPT_ORDER)
    n_models = len(model_names)

    data = np.zeros((n_scenarios, n_prompts * n_models))
    for i, sc in enumerate(SCENARIO_ORDER):
        for j, pr in enumerate(PROMPT_ORDER):
            for k, mn in enumerate(model_names):
                col = j * n_models + k
                data[i, col] = _get_metric(summary, sc, pr, mn, "hm_efficiency", 1.0)

    im = ax.imshow(data, cmap="RdYlGn", vmin=0.7, vmax=1.0, aspect="auto")
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels([s[:15] for s in SCENARIO_LABELS], fontsize=8)

    # X-axis: prompt labels with model sub-labels
    xtick_positions = []
    xtick_labels = []
    for j, pl in enumerate(PROMPT_LABELS):
        for k, mn in enumerate(model_names):
            col = j * n_models + k
            xtick_positions.append(col)
            label = f"{mn.split('-')[0]}" if k == 0 else f"{mn.split('-')[0]}"
            xtick_labels.append(label)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=6, rotation=45, ha="right")

    # Add prompt group labels
    for j, pl in enumerate(PROMPT_LABELS):
        center = j * n_models + (n_models - 1) / 2
        ax.text(center, -1.5, pl, ha="center", fontsize=7, fontweight="bold")

    # Add value annotations
    for i in range(n_scenarios):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if val < 0.85 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    ax.set_title("(a) HM Efficiency", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (b) SARP Pass/Fail Grid
    ax = axes[0, 1]
    sarp_data = np.zeros((n_scenarios, n_prompts * n_models))
    for i, sc in enumerate(SCENARIO_ORDER):
        for j, pr in enumerate(PROMPT_ORDER):
            for k, mn in enumerate(model_names):
                col = j * n_models + k
                sarp_data[i, col] = 1.0 if _get_metric(summary, sc, pr, mn, "is_sarp", True) else 0.0

    colors = np.zeros((*sarp_data.shape, 3))
    colors[sarp_data == 1.0] = [0.2, 0.7, 0.3]  # green
    colors[sarp_data == 0.0] = [0.8, 0.2, 0.2]  # red

    ax.imshow(colors, aspect="auto")
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels([s[:15] for s in SCENARIO_LABELS], fontsize=8)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=6, rotation=45, ha="right")

    for i in range(n_scenarios):
        for j in range(sarp_data.shape[1]):
            label = "PASS" if sarp_data[i, j] == 1.0 else "FAIL"
            ax.text(j, i, label, ha="center", va="center", fontsize=6,
                    color="white", fontweight="bold")

    ax.set_title("(b) SARP Consistency", fontsize=10, fontweight="bold")
    pass_patch = mpatches.Patch(color=[0.2, 0.7, 0.3], label="PASS")
    fail_patch = mpatches.Patch(color=[0.8, 0.2, 0.2], label="FAIL")
    ax.legend(handles=[pass_patch, fail_patch], fontsize=7, loc="upper right")

    # (c) Instinct vs Reasoning Scatter
    ax = axes[0, 2]
    if len(model_names) >= 2:
        m0, m1 = model_names[0], model_names[1]
        scatter_colors = plt.cm.tab10(np.linspace(0, 1, n_scenarios))

        for i, sc in enumerate(SCENARIO_ORDER):
            x_vals, y_vals = [], []
            for pr in PROMPT_ORDER:
                x = _get_metric(summary, sc, pr, m0, "hm_efficiency", 1.0)
                y = _get_metric(summary, sc, pr, m1, "hm_efficiency", 1.0)
                x_vals.append(x)
                y_vals.append(y)
            ax.scatter(x_vals, y_vals, c=[scatter_colors[i]], s=60, alpha=0.8,
                       label=SCENARIO_LABELS[i][:15], edgecolors="black", linewidths=0.5)

        # Diagonal
        ax.plot([0.5, 1.05], [0.5, 1.05], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel(f"{m0} HM Efficiency", fontsize=8)
        ax.set_ylabel(f"{m1} HM Efficiency", fontsize=8)
        ax.set_xlim(0.5, 1.05)
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=6, loc="lower right")
        ax.text(0.55, 1.0, f"{m1} more\nconsistent", fontsize=7, alpha=0.5)
        ax.text(0.9, 0.55, f"{m0} more\nconsistent", fontsize=7, alpha=0.5)
    ax.set_title("(c) Instinct vs Reasoning", fontsize=10, fontweight="bold")

    # (d) Violation Count Comparison
    ax = axes[1, 0]
    x_pos = np.arange(n_scenarios)
    width = 0.35
    for k, mn in enumerate(model_names):
        violations = []
        for sc in SCENARIO_ORDER:
            total = sum(
                _get_metric(summary, sc, pr, mn, "n_sarp_violations", 0)
                for pr in PROMPT_ORDER
            )
            violations.append(total)
        offset = (k - (n_models - 1) / 2) * width
        bars = ax.bar(x_pos + offset, violations, width, label=mn, alpha=0.8)
        for bar, v in zip(bars, violations):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(v), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([s[:12] for s in SCENARIO_LABELS], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Total SARP Violations", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_title("(d) Violations by Scenario", fontsize=10, fontweight="bold")

    # (e) Prompt Type Ranking
    ax = axes[1, 1]
    prompt_means = []
    for pr in PROMPT_ORDER:
        vals = []
        for sc in SCENARIO_ORDER:
            for mn in model_names:
                v = _get_metric(summary, sc, pr, mn, "hm_efficiency", 1.0)
                vals.append(v)
        prompt_means.append(np.mean(vals) if vals else 1.0)

    y_pos = np.arange(n_prompts)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_prompts))
    bars = ax.barh(y_pos, prompt_means, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(PROMPT_LABELS, fontsize=8)
    ax.set_xlabel("Mean HM Efficiency", fontsize=9)
    ax.set_xlim(min(prompt_means) - 0.05, 1.01)

    for bar, val in zip(bars, prompt_means):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.set_title("(e) Prompt Strategy Ranking", fontsize=10, fontweight="bold")

    # (f) Worst-case summary text (preference graph needs per-user data we don't have in summary)
    ax = axes[1, 2]
    ax.axis("off")

    # Find worst case
    worst_key = None
    worst_sc = None
    worst_violations = 0
    for sc, sresults in summary.items():
        for key, data in sresults.items():
            if data["n_sarp_violations"] > worst_violations:
                worst_violations = data["n_sarp_violations"]
                worst_key = key
                worst_sc = sc

    if worst_key and worst_sc:
        data = summary[worst_sc][worst_key]
        parts = worst_key.split("__")
        lines = [
            "WORST-CASE ANALYSIS",
            "",
            f"Scenario: {ALL_SCENARIOS[worst_sc].display_name}",
            f"Prompt: {parts[0]}",
            f"Model: {parts[1] if len(parts) > 1 else '?'}",
            "",
            f"SARP violations: {data['n_sarp_violations']}",
            f"WARP violations: {data['n_warp_violations']}",
            f"HM efficiency: {data['hm_efficiency']:.3f}",
            f"Max SCC size: {data['max_scc']}",
            f"Observations: {data['n_observations']}",
            "",
            "Choice distribution:",
        ]
        for item, count in sorted(data.get("choice_distribution", {}).items(),
                                    key=lambda x: -x[1]):
            pct = count / data["n_observations"] * 100
            lines.append(f"  {item}: {count} ({pct:.0f}%)")
    else:
        lines = ["No SARP violations found!", "", "All prompts and models", "are perfectly consistent."]

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("(f) Worst-Case Detail", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "app_llm_benchmark_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def main() -> None:
    summary = load_summary()
    generate_panel(summary)


if __name__ == "__main__":
    main()
