"""Print benchmark results in paper-ready format with per-dataset persistence."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from case_studies.benchmarks.core.evaluation import BenchmarkResult


def _slugify(name: str) -> str:
    """Convert dataset name to filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _fmt_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds <= 0:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{int(s)}s"


def _fmt_mem(mb: float) -> str:
    """Format peak memory in MB."""
    if mb <= 0:
        return "-"
    return f"{mb:.0f}MB"


# ---------------------------------------------------------------------------
# Per-dataset persistence
# ---------------------------------------------------------------------------


def save_dataset_results(
    results: list[BenchmarkResult], dataset_name: str, output_dir: Path
) -> Path:
    """Save results for a single dataset to its own JSON file."""
    slug = _slugify(dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"results_{slug}.json"
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"  Saved {len(results)} results to {path}")
    return path


def load_dataset_results(
    dataset_name: str, output_dir: Path
) -> list[BenchmarkResult] | None:
    """Load previously saved results for a single dataset. Returns None if not found."""
    slug = _slugify(dataset_name)
    path = output_dir / f"results_{slug}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return [BenchmarkResult(**d) for d in data]


def aggregate_all_results(output_dir: Path) -> list[BenchmarkResult]:
    """Read all per-dataset result files and merge into one list."""
    all_results: list[BenchmarkResult] = []
    for p in sorted(output_dir.glob("results_*.json")):
        if p.name == "results.json":
            continue
        with open(p) as f:
            data = json.load(f)
        all_results.extend(BenchmarkResult(**d) for d in data)
    return all_results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def print_summary(results: list[BenchmarkResult]) -> None:
    cls = [r for r in results if r.task_type == "classification"]
    reg = [r for r in results if r.task_type == "regression"]

    print("\n" + "=" * 160)
    print(" ML BENCHMARK: Do RP Graph Features Add Predictive Power?")
    print(" Protocol: Per-user temporal split | 80/20 user holdout | CatBoost defaults | Bootstrap CI")
    print("=" * 160)

    if cls:
        print(
            f"\n  {'Dataset':<20} {'Target':<20} {'N':>6} {'%pos':>5} "
            f"{'AP Base':>8} {'AP +RP':>8} {'Lift':>7} {'95% CI':>16} {'p-val':>6} "
            f"{'AUC Base':>9} {'AUC +RP':>8} "
            f"{'Load':>7} {'Engine':>8} {'Feat':>7} {'Mem':>6}"
        )
        print("  " + "-" * 156)

        last_dataset = None
        for r in cls:
            # Primary metric: always AUC-PR (average precision)
            base_m, comb_m = r.ap_base, r.ap_combined

            ci_str = f"({r.lift_ci_lower:+.1f}, {r.lift_ci_upper:+.1f})"
            sig = (
                "***" if r.lift_p_value < 0.01
                else "**" if r.lift_p_value < 0.05
                else "*" if r.lift_p_value < 0.1
                else ""
            )

            # AUC-ROC as supplementary column
            auc_base_str = f"{r.auc_base:.3f}"
            auc_comb_str = f"{r.auc_combined:.3f}"

            # Timing: only show on first row per dataset
            if r.dataset != last_dataset:
                load_str = _fmt_time(r.load_time_s)
                eng_str = _fmt_time(r.engine_time_s)
                feat_str = _fmt_time(r.feature_time_s)
                mem_str = _fmt_mem(r.peak_memory_mb)
                last_dataset = r.dataset
            else:
                load_str = eng_str = feat_str = mem_str = ""

            print(
                f"  {r.dataset:<20} {r.target:<20} {r.n_test:>6} {r.positive_rate:>5.1%} "
                f"{base_m:>8.3f} {comb_m:>8.3f} {r.lift_pct:>+6.1f}% {ci_str:>16} {r.lift_p_value:>5.3f}{sig:<3} "
                f"{auc_base_str:>9} {auc_comb_str:>8} "
                f"{load_str:>7} {eng_str:>8} {feat_str:>7} {mem_str:>6}"
            )

        print("  " + "-" * 156)
        print("  Primary metric: AUC-PR (average precision). AUC-ROC shown as supplementary.")
        print("  Lift % and bootstrap CI are computed on AUC-PR.")

    if reg:
        print(
            f"\n  {'Dataset':<20} {'Target':<20} {'N':>6} "
            f"{'Base R2':>8} {'+RP R2':>8} {'dR2':>7} "
            f"{'Load':>7} {'Engine':>8} {'Feat':>7} {'Mem':>6}"
        )
        print("  " + "-" * 100)

        last_dataset = None
        for r in reg:
            delta = r.r2_combined - r.r2_base

            if r.dataset != last_dataset:
                load_str = _fmt_time(r.load_time_s)
                eng_str = _fmt_time(r.engine_time_s)
                feat_str = _fmt_time(r.feature_time_s)
                mem_str = _fmt_mem(r.peak_memory_mb)
                last_dataset = r.dataset
            else:
                load_str = eng_str = feat_str = mem_str = ""

            print(
                f"  {r.dataset:<20} {r.target:<20} {r.n_test:>6} "
                f"{r.r2_base:>8.3f} {r.r2_combined:>8.3f} {delta:>+7.3f} "
                f"{load_str:>7} {eng_str:>8} {feat_str:>7} {mem_str:>6}"
            )
        print("  " + "-" * 100)

    total_time = sum(r.wall_time_s for r in results)
    print(f"\n  Wall time: {total_time:.0f}s")
    print()


# ---------------------------------------------------------------------------
# Combined save (backward compat)
# ---------------------------------------------------------------------------


def save_results(results: list[BenchmarkResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"  Combined results saved to {output_dir}/results.json")


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def generate_plots(results: list[BenchmarkResult], output_dir: Path) -> None:
    """Generate benchmark visualization plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed, skipping plots")
        return

    if not results:
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _plot_lift_comparison(results, fig_dir, plt)
    _plot_auc_comparison(results, fig_dir, plt)
    _plot_timing_breakdown(results, fig_dir, plt)
    _plot_peak_memory(results, fig_dir, plt)

    print(f"  Plots saved to {fig_dir}/")


def _plot_lift_comparison(results, fig_dir, plt):
    """Horizontal bar chart of lift % with CI error bars."""
    cls = [r for r in results if r.task_type == "classification"]
    if not cls:
        return

    labels = [f"{r.dataset}\n{r.target}" for r in cls]
    lifts = [r.lift_pct for r in cls]
    errs_lo = [r.lift_pct - r.lift_ci_lower for r in cls]
    errs_hi = [r.lift_ci_upper - r.lift_pct for r in cls]

    fig, ax = plt.subplots(figsize=(10, max(4, len(cls) * 0.5)))
    y_pos = range(len(cls))
    ax.barh(y_pos, lifts, xerr=[errs_lo, errs_hi], capsize=3, color="#4C72B0", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lift %")
    ax.set_title("Lift from Adding RP Features (with 95% CI)")
    plt.tight_layout()
    fig.savefig(fig_dir / "lift_comparison.png", dpi=150)
    plt.close(fig)


def _plot_auc_comparison(results, fig_dir, plt):
    """Grouped bar chart: RP-only, Baseline, Combined per target (AUC-PR)."""
    cls = [r for r in results if r.task_type == "classification"]
    if not cls:
        return

    labels = [f"{r.dataset}\n{r.target}" for r in cls]
    rp_only = [r.ap_rp for r in cls]
    baseline = [r.ap_base for r in cls]
    combined = [r.ap_combined for r in cls]

    x = np.arange(len(cls))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(cls) * 1.2), 6))
    ax.bar(x - width, rp_only, width, label="RP-only", color="#4C72B0", alpha=0.8)
    ax.bar(x, baseline, width, label="Baseline", color="#DD8452", alpha=0.8)
    ax.bar(x + width, combined, width, label="Combined", color="#55A868", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("AUC-PR (Average Precision)")
    ax.set_title("Model Comparison: RP-only vs Baseline vs Combined")
    ax.legend()
    ax.set_ylim(bottom=max(0, min(rp_only + baseline + combined) - 0.05))
    plt.tight_layout()
    fig.savefig(fig_dir / "auc_comparison.png", dpi=150)
    plt.close(fig)


def _plot_timing_breakdown(results, fig_dir, plt):
    """Stacked bar chart of load/engine/feature time per dataset."""
    # Deduplicate: one timing entry per dataset
    seen = {}
    for r in results:
        if r.dataset not in seen:
            seen[r.dataset] = r

    datasets = list(seen.keys())
    load_times = [seen[d].load_time_s for d in datasets]
    engine_times = [seen[d].engine_time_s for d in datasets]
    feature_times = [
        max(0, seen[d].feature_time_s - seen[d].engine_time_s) for d in datasets
    ]

    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.2), 5))
    ax.bar(x, load_times, label="Load", color="#4C72B0", alpha=0.8)
    ax.bar(x, engine_times, bottom=load_times, label="Engine (Rust)", color="#DD8452", alpha=0.8)
    bottoms = [l + e for l, e in zip(load_times, engine_times)]
    ax.bar(x, feature_times, bottom=bottoms, label="Features (Python)", color="#55A868", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Pipeline Timing Breakdown by Dataset")
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "timing_breakdown.png", dpi=150)
    plt.close(fig)


def _plot_peak_memory(results, fig_dir, plt):
    """Bar chart of peak memory per dataset."""
    seen = {}
    for r in results:
        if r.dataset not in seen:
            seen[r.dataset] = r

    datasets = list(seen.keys())
    mems = [seen[d].peak_memory_mb for d in datasets]

    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.0), 4))
    ax.bar(datasets, mems, color="#4C72B0", alpha=0.8)
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak Memory Usage by Dataset")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_dir / "peak_memory.png", dpi=150)
    plt.close(fig)
