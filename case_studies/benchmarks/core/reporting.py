"""Print benchmark results in paper-ready format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from case_studies.benchmarks.core.evaluation import BenchmarkResult


def print_summary(results: list[BenchmarkResult]) -> None:
    cls = [r for r in results if r.task_type == "classification"]
    reg = [r for r in results if r.task_type == "regression"]

    print("\n" + "=" * 110)
    print(" ML BENCHMARK: Do RP Graph Features Add Predictive Power?")
    print(" Protocol: Per-user temporal split | 80/20 user holdout | CatBoost defaults | Bootstrap CI")
    print("=" * 110)

    if cls:
        print(f"\n  {'Dataset':<16} {'Target':<16} {'N':>5} {'%pos':>5} "
              f"{'Baseline':>9} {'Combined':>9} {'Lift':>8} {'95% CI':>16} {'p-val':>6} "
              f"{'RP Group':>9} {'Base Group':>10}")
        print("  " + "-" * 108)
        for r in cls:
            # Use AP for imbalanced, AUC for balanced
            if r.positive_rate < 0.15:
                base_m, comb_m, metric = r.ap_base, r.ap_combined, "AP"
            else:
                base_m, comb_m, metric = r.auc_base, r.auc_combined, "AUC"

            rp_drop = r.group_importance.get("RP_features", 0)
            base_drop = r.group_importance.get("Baseline_features", 0)
            ci_str = f"({r.lift_ci_lower:+.1f}, {r.lift_ci_upper:+.1f})"
            sig = "***" if r.lift_p_value < 0.01 else "**" if r.lift_p_value < 0.05 else "*" if r.lift_p_value < 0.1 else ""

            print(f"  {r.dataset:<16} {r.target:<16} {r.n_test:>5} {r.positive_rate:>5.1%} "
                  f"{base_m:>9.3f} {comb_m:>9.3f} {r.lift_pct:>+7.1f}% {ci_str:>16} {r.lift_p_value:>5.3f}{sig} "
                  f"{rp_drop:>9.4f} {base_drop:>10.4f}")
        print("  " + "-" * 108)
        print("  Metric: AUC-PR for targets with <15% positive rate, AUC-ROC otherwise")
        print("  Group importance: drop in AUC when shuffling all features in the group (higher = more important)")

    if reg:
        print(f"\n  {'Dataset':<16} {'Target':<18} {'N':>5} "
              f"{'Base R²':>8} {'+RP R²':>8} {'Lift':>8} {'95% CI':>16} {'p-val':>6}")
        print("  " + "-" * 85)
        for r in reg:
            ci_str = f"({r.lift_ci_lower:+.1f}, {r.lift_ci_upper:+.1f})"
            print(f"  {r.dataset:<16} {r.target:<18} {r.n_test:>5} "
                  f"{r.r2_base:>8.3f} {r.r2_combined:>8.3f} {r.lift_pct:>+7.1f}% {ci_str:>16} {r.lift_p_value:>5.3f}")
        print("  " + "-" * 85)

    total_time = sum(r.wall_time_s for r in results)
    print(f"\n  Wall time: {total_time:.0f}s")
    print()


def save_results(results: list[BenchmarkResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"  Results saved to {output_dir}/")


def generate_plots(results: list[BenchmarkResult], output_dir: Path) -> None:
    pass
