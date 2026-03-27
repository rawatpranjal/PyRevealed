"""Dunnhumby grocery benchmark: churn, high-spender, spend-change prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import TRAIN_FRACTION, MIN_OBS_BUDGET, MIN_TRAIN_BUDGET, MIN_TEST_BUDGET
from case_studies.benchmarks.core.features import extract_budget_baseline, extract_budget_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "Dunnhumby"
DEFAULT_DATA_DIR = str(Path(__file__).resolve().parents[2] / "dunnhumby" / "data")


def load_and_prepare(data_dir=None, n_households=None):
    """Load Dunnhumby and prepare train/target splits.

    Returns:
        Tuple of (X_rp, X_base, targets_dict, user_ids)
        where targets_dict maps target_name -> (y_array, task_type)
    """
    from pyrevealed.datasets import load_dunnhumby

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    panel = load_dunnhumby(data_dir=data_dir, n_households=n_households, min_weeks=MIN_OBS_BUDGET)

    user_ids = []
    train_tuples = []
    train_mean_spends = []
    test_mean_spends = []
    test_total_spends = []

    for uid, log in panel._logs.items():
        T = log.num_records
        if T < MIN_OBS_BUDGET:
            continue

        split = int(T * TRAIN_FRACTION)
        if split < MIN_TRAIN_BUDGET or (T - split) < MIN_TEST_BUDGET:
            continue

        prices_train = log.cost_vectors[:split]
        qty_train = log.action_vectors[:split]
        prices_test = log.cost_vectors[split:]
        qty_test = log.action_vectors[split:]

        train_tuples.append((prices_train, qty_train))
        user_ids.append(uid)

        train_spend_per_obs = np.sum(prices_train * qty_train, axis=1)
        test_spend_per_obs = np.sum(prices_test * qty_test, axis=1)

        train_mean_spends.append(float(np.mean(train_spend_per_obs)))
        test_mean_spends.append(float(np.mean(test_spend_per_obs)))
        test_total_spends.append(float(np.sum(test_spend_per_obs)))

    train_mean_spends = np.array(train_mean_spends)
    test_mean_spends = np.array(test_mean_spends)
    test_total_spends = np.array(test_total_spends)

    print(f"  Users: {len(user_ids)}")

    # Extract features
    print(f"  Extracting baseline features...")
    X_base = extract_budget_baseline(train_tuples, user_ids)

    print(f"  Extracting RP features via Engine...")
    X_rp_engine = extract_budget_rp(train_tuples, user_ids)

    X_rp = X_rp_engine

    # --- Targets ---

    # Churn: mean spend dropped by >50% from train to test window
    spend_ratio = test_mean_spends / np.maximum(train_mean_spends, 1e-6)
    churn = (spend_ratio < 0.5).astype(int)

    # High spender: top tercile of test-window total spend
    threshold = np.percentile(test_total_spends, 66.67)
    high_spender = (test_total_spends > threshold).astype(int)

    # Spend change: difference in mean spend (regression)
    spend_change = test_mean_spends - train_mean_spends

    targets_dict = {
        "Churn": (churn, "classification"),
        "High Spender": (high_spender, "classification"),
        "Spend Change": (spend_change, "regression"),
    }

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, n_households=None) -> list[BenchmarkResult]:
    """Run all Dunnhumby benchmarks."""
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
                  f"Combined={result.auc_combined:.3f}  Lift={result.auc_lift:+.3f}")
        else:
            print(f"    R2: RP={result.r2_rp:.3f}  Base={result.r2_base:.3f}  "
                  f"Combined={result.r2_combined:.3f}")

    return results
