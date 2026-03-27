"""H&M Fashion benchmark: high spender, churn, spend change.

1.36M customers, 31.8M transactions (2018-2020). Budget-based with real prices.
Aggregated to monthly periods by product group.

NOTE: H&M has REAL transaction prices — normalized but genuine price variation.
This is the largest dataset in the benchmark suite by transaction volume.

ASSUMPTION AUDIT:
- Prices: Real (from transactions_train.csv 'price' column, normalized)
- Quantities: Actual item counts per product group per month
- Categories: Product groups derived from article_id prefix (top 20)
- No synthetic/imputed data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import TRAIN_FRACTION, MIN_OBS_BUDGET
from case_studies.benchmarks.core.features import extract_budget_baseline, extract_budget_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "H&M"


def load_and_prepare(data_dir=None, max_users=50000):
    """Load H&M with temporal split."""
    from pyrevealed.datasets._hm import load_hm

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    panel = load_hm(data_dir=data_dir, max_users=max_users, min_months=6)

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
        if split < 4 or (T - split) < 2:
            continue

        prices_train = log.cost_vectors[:split]
        qty_train = log.action_vectors[:split]
        prices_test = log.cost_vectors[split:]
        qty_test = log.action_vectors[split:]

        train_tuples.append((prices_train, qty_train))
        user_ids.append(uid)

        train_spend = np.sum(prices_train * qty_train, axis=1)
        test_spend = np.sum(prices_test * qty_test, axis=1)
        train_mean_spends.append(float(np.mean(train_spend)))
        test_mean_spends.append(float(np.mean(test_spend)))
        test_total_spends.append(float(np.sum(test_spend)))

    train_mean_spends = np.array(train_mean_spends)
    test_mean_spends = np.array(test_mean_spends)
    test_total_spends = np.array(test_total_spends)

    print(f"  Users: {len(user_ids)}")

    if len(user_ids) < 30:
        return None, None, {}, user_ids

    print(f"  Extracting baseline features...")
    X_base = extract_budget_baseline(train_tuples, user_ids)

    print(f"  Extracting RP features via Engine...")
    X_rp = extract_budget_rp(train_tuples, user_ids)

    # Targets
    threshold = np.percentile(test_total_spends, 66.67)
    high_spender = (test_total_spends > threshold).astype(int)

    spend_ratio = test_mean_spends / np.maximum(train_mean_spends, 1e-6)
    churn = (spend_ratio < 0.5).astype(int)

    spend_change = test_mean_spends - train_mean_spends

    targets_dict = {
        "High Spender": (high_spender, "classification"),
        "Churn": (churn, "classification"),
        "Spend Change": (spend_change, "regression"),
    }

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, max_users=50000) -> list[BenchmarkResult]:
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, max_users)
    if X_rp is None:
        return []

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
