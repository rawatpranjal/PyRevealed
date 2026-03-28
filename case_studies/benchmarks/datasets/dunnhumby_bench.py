"""Dunnhumby grocery benchmark: global calendar cutoff + user holdout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import MIN_OBS_BUDGET
from case_studies.benchmarks.core.features import extract_budget_baseline, extract_budget_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "Dunnhumby"
DEFAULT_DATA_DIR = str(Path(__file__).resolve().parents[2] / "dunnhumby" / "data")


def load_and_prepare(data_dir=None, n_households=None):
    """Load Dunnhumby with GLOBAL calendar cutoff.

    Instead of per-user 70/30 split, finds the global 70th percentile
    observation index and splits ALL users at that point. Users who
    have no data after the cutoff are correctly labeled as churned
    (test_mean_spend = 0).
    """
    from prefgraph.datasets import load_dunnhumby

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    panel = load_dunnhumby(data_dir=data_dir, n_households=n_households, min_weeks=3)

    # Step 1: Find global cutoff — 70th percentile of observation counts
    all_T = [log.num_records for log in panel._logs.values()]
    global_cutoff = int(np.percentile(all_T, 70))
    print(f"  Global cutoff: observation {global_cutoff} (70th pctl of T across {len(all_T)} users)")

    # Step 2: Split each user at the SAME global cutoff
    user_ids = []
    train_tuples = []
    train_mean_spends = []
    test_mean_spends = []
    test_total_spends = []

    for uid, log in panel._logs.items():
        T = log.num_records
        # Need at least MIN_OBS_BUDGET observations in train window
        train_end = min(global_cutoff, T)
        if train_end < MIN_OBS_BUDGET:
            continue

        prices_train = log.cost_vectors[:train_end]
        qty_train = log.action_vectors[:train_end]
        train_tuples.append((prices_train, qty_train))
        user_ids.append(uid)

        train_spend = np.sum(prices_train * qty_train, axis=1)
        train_mean_spends.append(float(np.mean(train_spend)))

        # Users with NO data after cutoff → test_mean_spend = 0 (churned)
        if T > global_cutoff:
            prices_test = log.cost_vectors[global_cutoff:]
            qty_test = log.action_vectors[global_cutoff:]
            test_spend = np.sum(prices_test * qty_test, axis=1)
            test_mean_spends.append(float(np.mean(test_spend)))
            test_total_spends.append(float(np.sum(test_spend)))
        else:
            test_mean_spends.append(0.0)
            test_total_spends.append(0.0)

    train_mean_spends = np.array(train_mean_spends)
    test_mean_spends = np.array(test_mean_spends)
    test_total_spends = np.array(test_total_spends)

    n_churned = int(np.sum(test_total_spends == 0))
    print(f"  Users: {len(user_ids)} ({n_churned} churned — zero future spend)")

    # Features from PAST only
    print(f"  Extracting baseline features...")
    X_base = extract_budget_baseline(train_tuples, user_ids)

    print(f"  Extracting RP features via Engine...")
    X_rp = extract_budget_rp(train_tuples, user_ids)

    # Targets from FUTURE only
    spend_ratio = test_mean_spends / np.maximum(train_mean_spends, 1e-6)
    churn = (spend_ratio < 0.5).astype(int)

    threshold = np.percentile(test_total_spends[test_total_spends > 0], 66.67)
    high_spender = (test_total_spends > threshold).astype(int)

    spend_change = test_mean_spends - train_mean_spends

    targets_dict = {
        "Churn": (churn, "classification"),
        "High Spender": (high_spender, "classification"),
        "Spend Change": (spend_change, "regression"),
    }

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, n_households=None) -> list[BenchmarkResult]:
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, n_households)

    results = []
    for target_name, (y, task_type) in targets_dict.items():
        print(f"  [{DATASET_NAME}] Target: {target_name} ({task_type})")
        if task_type == "classification":
            pos_rate = np.mean(y)
            if pos_rate < 0.02 or pos_rate > 0.98:
                print(f"    Skipping — too imbalanced (pos_rate={pos_rate:.3f})")
                continue

        result = run_three_way(X_rp, X_base, y, DATASET_NAME, target_name, task_type)
        results.append(result)

        if task_type == "classification":
            print(f"    AUC: RP={result.auc_rp:.3f}  Base={result.auc_base:.3f}  "
                  f"Combined={result.auc_combined:.3f}  Lift={result.auc_combined - result.auc_base:+.3f}")
        else:
            print(f"    R2: RP={result.r2_rp:.3f}  Base={result.r2_base:.3f}  "
                  f"Combined={result.r2_combined:.3f}")

    return results
