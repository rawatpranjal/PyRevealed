"""Generate summary tables and plots from benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from case_studies.benchmarks.core.evaluation import BenchmarkResult


def results_to_summary_table(results: list[BenchmarkResult]) -> pd.DataFrame:
    """Create a summary table from benchmark results.

    Returns a DataFrame with one row per dataset × target.
    """
    rows = []
    for r in results:
        row = {
            "Dataset": r.dataset,
            "Target": r.target,
            "N": r.n_users,
            "Task": r.task_type[:5],
        }
        if r.task_type == "classification":
            pct_lift = (r.auc_combined - r.auc_base) / max(r.auc_base, 0.5) * 100 if r.auc_base > 0 else 0
            row.update({
                "AUC (RP only)": f"{r.auc_rp:.3f}",
                "AUC (Baseline)": f"{r.auc_base:.3f}",
                "AUC (Combined)": f"{r.auc_combined:.3f}",
                "Lift %": f"{pct_lift:+.1f}%",
            })
        else:
            row.update({
                "RMSE (RP only)": f"{r.rmse_rp:.4f}",
                "RMSE (Baseline)": f"{r.rmse_base:.4f}",
                "RMSE (Combined)": f"{r.rmse_combined:.4f}",
                "R2 (Combined)": f"{r.r2_combined:.3f}",
            })
        rows.append(row)

    return pd.DataFrame(rows)


def save_results(results: list[BenchmarkResult], output_dir: Path) -> None:
    """Save results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    # CSV summary
    table = results_to_summary_table(results)
    table.to_csv(output_dir / "summary_table.csv", index=False)

    print(f"\nResults saved to {output_dir}/")


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a formatted summary table to stdout."""
    cls_results = [r for r in results if r.task_type == "classification"]
    reg_results = [r for r in results if r.task_type == "regression"]
    total_n = sum(r.n_users for r in results)

    print("\n" + "=" * 90)
    print(" ML BENCHMARK: RP Features as Predictive Signals")
    print("=" * 90)
    print(f"\n  {len(results)} tasks across {len(set(r.dataset for r in results))} datasets, "
          f"{total_n:,} total users.")
    if cls_results:
        lifts = [(r.auc_combined - r.auc_base) / max(r.auc_base, 0.5) * 100
                 for r in cls_results if r.auc_base > 0.5]
        if lifts:
            print(f"  Marginal lift from RP features: {min(lifts):+.1f}% to {max(lifts):+.1f}% AUC.")
    print()

    # Main table: focus on Baseline → Combined → Lift
    if cls_results:
        print("  Classification (AUC-ROC, 5-fold Stratified CV)")
        print("  " + "-" * 88)
        print(f"  {'Dataset':<18} {'Target':<18} {'N':>6} {'%pos':>5}  {'Baseline':>10}  {'+RP':>10}  {'Lift%':>7}  {'AUC-PR':>7}")
        print("  " + "-" * 88)
        for r in cls_results:
            pct_lift = (r.auc_combined - r.auc_base) / max(r.auc_base, 0.5) * 100 if r.auc_base > 0.5 else 0.0
            print(
                f"  {r.dataset:<18} {r.target:<18} {r.n_users:>6} {r.positive_rate:>5.1%}  "
                f"{r.auc_base:>.3f}±{r.auc_base_std:.3f}  {r.auc_combined:>.3f}±{r.auc_combined_std:.3f}  "
                f"{pct_lift:>+6.1f}%  {r.ap_combined:>7.3f}"
            )
        print("  " + "-" * 88)

        print("\n  Classification Tasks — AUC-PR (Average Precision, better for imbalanced targets)")
        print("  " + "-" * 100)
        print(f"  {'Dataset':<18} {'Target':<18} {'N':>6} {'%pos':>5}  {'RP only':>10}  {'Baseline':>10}  {'Combined':>10}")
        print("  " + "-" * 100)
        for r in cls_results:
            print(
                f"  {r.dataset:<18} {r.target:<18} {r.n_users:>6} {r.positive_rate:>5.1%}  "
                f"{r.ap_rp:>10.3f}  {r.ap_base:>10.3f}  {r.ap_combined:>10.3f}"
            )
        print("  " + "-" * 100)

        print("\n  Classification Tasks — Log Loss (lower = better calibration)")
        print("  " + "-" * 100)
        print(f"  {'Dataset':<18} {'Target':<18}  {'RP only':>10}  {'Baseline':>10}  {'Combined':>10}")
        print("  " + "-" * 100)
        for r in cls_results:
            print(
                f"  {r.dataset:<18} {r.target:<18}  "
                f"{r.logloss_rp:>10.4f}  {r.logloss_base:>10.4f}  {r.logloss_combined:>10.4f}"
            )
        print("  " + "-" * 100)

        print("\n  Classification Tasks — In-Sample AUC-ROC (overfitting check)")
        print("  " + "-" * 80)
        print(f"  {'Dataset':<18} {'Target':<18}  {'RP only':>10}  {'Baseline':>10}  {'Combined':>10}")
        print("  " + "-" * 80)
        for r in cls_results:
            print(
                f"  {r.dataset:<18} {r.target:<18}  "
                f"{r.auc_rp_train:>10.3f}  {r.auc_base_train:>10.3f}  {r.auc_combined_train:>10.3f}"
            )
        print("  " + "-" * 80)

    # Regression results
    reg_results = [r for r in results if r.task_type == "regression"]
    if reg_results:
        print("\n  Regression Tasks — Out-of-Sample R² (5-fold CV)")
        print("  " + "-" * 86)
        print(f"  {'Dataset':<18} {'Target':<22} {'N':>6}  {'RP only':>8}  {'Baseline':>8}  {'Combined':>8}")
        print("  " + "-" * 86)
        for r in reg_results:
            print(
                f"  {r.dataset:<18} {r.target:<22} {r.n_users:>6}  "
                f"{r.r2_rp:>8.3f}  {r.r2_base:>8.3f}  {r.r2_combined:>8.3f}"
            )
        print("  " + "-" * 86)

        print("\n  Regression Tasks — In-Sample R² (overfitting check)")
        print("  " + "-" * 86)
        print(f"  {'Dataset':<18} {'Target':<22}  {'RP only':>8}  {'Baseline':>8}  {'Combined':>8}")
        print("  " + "-" * 86)
        for r in reg_results:
            print(
                f"  {r.dataset:<18} {r.target:<22}  "
                f"{r.r2_rp_train:>8.3f}  {r.r2_base_train:>8.3f}  {r.r2_combined_train:>8.3f}"
            )
        print("  " + "-" * 86)

    # Timing
    timed_results = [r for r in results if r.wall_time_s > 0]
    if timed_results:
        print("\n  Execution Time (model training only, excludes data loading)")
        print("  " + "-" * 60)
        print(f"  {'Dataset':<18} {'Target':<18} {'N':>6}  {'Time':>8}")
        print("  " + "-" * 60)
        for r in timed_results:
            print(f"  {r.dataset:<18} {r.target:<18} {r.n_users:>6}  {r.wall_time_s:>7.1f}s")
        total_model_time = sum(r.wall_time_s for r in timed_results)
        print("  " + "-" * 60)
        print(f"  {'Total':<18} {'':<18} {sum(r.n_users for r in results):>6}  {total_model_time:>7.1f}s")
        print("  " + "-" * 60)

    # Feature importance summary
    combined_cls = [r for r in cls_results if r.top_features]
    if combined_cls:
        print("\n  Top Features (Combined Model, across classification tasks)")
        print("  " + "-" * 60)
        # Aggregate feature importance across all classification results
        feat_scores: dict[str, list[float]] = {}
        for r in combined_cls:
            for fname, score in r.top_features:
                feat_scores.setdefault(fname, []).append(score)
        avg_scores = {k: np.mean(v) for k, v in feat_scores.items()}
        top = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (fname, score) in enumerate(top, 1):
            print(f"    {i:2}. {fname:<30} {score:>8.1f}")
        print("  " + "-" * 60)

    print()


def generate_plots(results: list[BenchmarkResult], output_dir: Path) -> None:
    """Generate benchmark visualization plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cls_results = [r for r in results if r.task_type == "classification"]
    if not cls_results:
        return

    # --- Plot 1: Three-way AUC comparison bar chart ---
    fig, ax = plt.subplots(figsize=(max(10, len(cls_results) * 2.5), 6))

    labels = [f"{r.dataset}\n{r.target}" for r in cls_results]
    x = np.arange(len(labels))
    width = 0.25

    bars_rp = [r.auc_rp for r in cls_results]
    bars_base = [r.auc_base for r in cls_results]
    bars_comb = [r.auc_combined for r in cls_results]

    ax.bar(x - width, bars_rp, width, label="RP only", color="#2196F3", alpha=0.85)
    ax.bar(x, bars_base, width, label="Baseline only", color="#9E9E9E", alpha=0.85)
    ax.bar(x + width, bars_comb, width, label="RP + Baseline", color="#4CAF50", alpha=0.85)

    ax.set_ylabel("AUC-ROC")
    ax.set_title("Three-Way Comparison: RP Features as Predictive Signals")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.set_ylim(0.45, 1.0)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="Random")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(fig_dir / "auc_comparison.png", dpi=150)
    plt.close()

    # --- Plot 2: Feature importance from the first classification result ---
    if cls_results[0].top_features:
        r = cls_results[0]
        names = [f[0] for f in r.top_features[:12]]
        values = [f[1] for f in r.top_features[:12]]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2196F3" if any(kw in n for kw in ["ccei", "mpi", "garp", "harp", "hm", "vei", "sarp", "warp", "scc", "violation"]) else "#9E9E9E" for n in names]
        ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance (LightGBM)")
        ax.set_title(f"Feature Importance: {r.dataset} — {r.target}")
        ax.grid(True, alpha=0.2, axis="x")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#2196F3", label="RP feature"), Patch(facecolor="#9E9E9E", label="Baseline feature")]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        plt.savefig(fig_dir / "feature_importance.png", dpi=150)
        plt.close()

    print(f"  Plots saved to {fig_dir}/")
