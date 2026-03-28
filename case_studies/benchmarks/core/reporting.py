"""Generate summary tables from benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from case_studies.benchmarks.core.evaluation import BenchmarkResult, _compute_lift


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print OOS + OOT results side by side."""
    cls = [r for r in results if r.task_type == "classification"]
    reg = [r for r in results if r.task_type == "regression"]
    total_n = sum(r.n_users for r in results)

    print("\n" + "=" * 100)
    print(" ML BENCHMARK: RP Features as Predictive Signals (LightGBM defaults)")
    print("=" * 100)
    print(f"\n  {len(results)} tasks, {len(set(r.dataset for r in results))} datasets, "
          f"{total_n:,} total users.")

    if cls:
        print("\n  Classification — AUC-ROC")
        print("  " + "-" * 96)
        print(f"  {'Dataset':<16} {'Target':<16} {'N':>6} {'%pos':>5}  "
              f"{'OOS Base':>9} {'OOS +RP':>9} {'OOS Lift':>8}  "
              f"{'OOT Base':>9} {'OOT +RP':>9} {'OOT Lift':>8}")
        print("  " + "-" * 96)
        for r in cls:
            oos_lift = _compute_lift(r.oos_auc_combined, r.oos_auc_base)
            oot_lift = _compute_lift(r.oot_auc_combined, r.oot_auc_base)
            print(f"  {r.dataset:<16} {r.target:<16} {r.n_users:>6} {r.positive_rate:>5.1%}  "
                  f"{r.oos_auc_base:>9.3f} {r.oos_auc_combined:>9.3f} {oos_lift:>+7.1f}%  "
                  f"{r.oot_auc_base:>9.3f} {r.oot_auc_combined:>9.3f} {oot_lift:>+7.1f}%")
        print("  " + "-" * 96)

        print("\n  Classification — AUC-PR (better for imbalanced)")
        print("  " + "-" * 96)
        print(f"  {'Dataset':<16} {'Target':<16} {'%pos':>5}  "
              f"{'OOS Base':>9} {'OOS +RP':>9} {'OOS Lift':>8}  "
              f"{'OOT Base':>9} {'OOT +RP':>9} {'OOT Lift':>8}")
        print("  " + "-" * 96)
        for r in cls:
            oos_lift = (r.oos_ap_combined - r.oos_ap_base) / max(r.oos_ap_base, 0.01) * 100
            oot_lift = (r.oot_ap_combined - r.oot_ap_base) / max(r.oot_ap_base, 0.01) * 100
            print(f"  {r.dataset:<16} {r.target:<16} {r.positive_rate:>5.1%}  "
                  f"{r.oos_ap_base:>9.3f} {r.oos_ap_combined:>9.3f} {oos_lift:>+7.1f}%  "
                  f"{r.oot_ap_base:>9.3f} {r.oot_ap_combined:>9.3f} {oot_lift:>+7.1f}%")
        print("  " + "-" * 96)

    if reg:
        print("\n  Regression — R²")
        print("  " + "-" * 86)
        print(f"  {'Dataset':<16} {'Target':<18} {'N':>6}  "
              f"{'OOS Base':>9} {'OOS +RP':>9}  {'OOT Base':>9} {'OOT +RP':>9}")
        print("  " + "-" * 86)
        for r in reg:
            print(f"  {r.dataset:<16} {r.target:<18} {r.n_users:>6}  "
                  f"{r.oos_r2_base:>9.3f} {r.oos_r2_combined:>9.3f}  "
                  f"{r.oot_r2_base:>9.3f} {r.oot_r2_combined:>9.3f}")
        print("  " + "-" * 86)

    # Timing
    timed = [r for r in results if r.wall_time_s > 0]
    if timed:
        print(f"\n  Total time: {sum(r.wall_time_s for r in timed):.0f}s")

    # Top features
    combined_cls = [r for r in cls if r.top_features]
    if combined_cls:
        print("\n  Top Features (OOS Combined, LightGBM importance)")
        print("  " + "-" * 50)
        feat_scores: dict[str, list[float]] = {}
        for r in combined_cls:
            for fname, score in r.top_features:
                feat_scores.setdefault(fname, []).append(score)
        avg = {k: np.mean(v) for k, v in feat_scores.items()}
        top = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (fname, score) in enumerate(top, 1):
            print(f"    {i:2}. {fname:<30} {score:>8.1f}")
        print("  " + "-" * 50)

    print()


def save_results(results: list[BenchmarkResult], output_dir: Path) -> None:
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"\n  Results saved to {output_dir}/")


def generate_plots(results: list[BenchmarkResult], output_dir: Path) -> None:
    """Placeholder — plots can be added later."""
    pass
