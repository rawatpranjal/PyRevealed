"""Pakistan E-Commerce benchmark: high spender, churn, spend change.

115K customers, 1M+ transactions, 16 categories, REAL PRICES.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import TRAIN_FRACTION, MIN_OBS_BUDGET
from case_studies.benchmarks.core.features import extract_budget_baseline, extract_budget_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult

DATASET_NAME = "Pakistan"


def load_and_prepare(data_dir=None, max_users=50000):
    from pyrevealed.datasets._pakistan import load_pakistan

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    panel = load_pakistan(data_dir=data_dir, max_users=max_users, min_months=3)

    user_ids, train_tuples = [], []
    train_mean_spends, test_mean_spends, test_total_spends = [], [], []

    for uid, log in panel._logs.items():
        T = log.num_records
        if T < 4:
            continue
        split = int(T * TRAIN_FRACTION)
        if split < 2 or (T - split) < 2:
            continue

        p_tr, q_tr = log.cost_vectors[:split], log.action_vectors[:split]
        p_te, q_te = log.cost_vectors[split:], log.action_vectors[split:]
        train_tuples.append((p_tr, q_tr))
        user_ids.append(uid)

        tr_s = np.sum(p_tr * q_tr, axis=1)
        te_s = np.sum(p_te * q_te, axis=1)
        train_mean_spends.append(float(np.mean(tr_s)))
        test_mean_spends.append(float(np.mean(te_s)))
        test_total_spends.append(float(np.sum(te_s)))

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
