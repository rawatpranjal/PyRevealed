"""M5 Walmart Forecasting benchmark: demand prediction, consistency scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import TRAIN_FRACTION
from case_studies.benchmarks.core.features import extract_budget_baseline, extract_budget_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "M5 Walmart"


def load_and_prepare(data_dir=None, max_users=None):
    """Load M5 and prepare train/target splits.

    M5 has 10 stores × 7 departments = 70 "users" with ~277 weeks each.
    This is a large-T, small-N dataset ideal for temporal RP analysis.
    """
    from prefgraph.datasets._m5 import load_m5

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    panel = load_m5(data_dir=data_dir, aggregation="store_dept")

    user_ids = []
    train_tuples = []
    train_mean_spends = []
    test_mean_spends = []
    test_total_spends = []

    for uid, log in panel._logs.items():
        T = log.num_records
        if T < 20:
            continue

        split = int(T * TRAIN_FRACTION)
        if split < 10 or (T - split) < 5:
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

    print(f"  Users (store-dept combos): {len(user_ids)}")

    if len(user_ids) < 10:
        print(f"  Too few users, skipping.")
        return None, None, {}, user_ids

    print(f"  Extracting baseline features...")
    X_base = extract_budget_baseline(train_tuples, user_ids)

    print(f"  Extracting RP features via Engine...")
    X_rp = extract_budget_rp(train_tuples, user_ids)

    # Targets
    threshold = np.percentile(test_total_spends, 66.67)
    high_demand = (test_total_spends > threshold).astype(int)

    spend_change = test_mean_spends - train_mean_spends

    # Demand volatility: std of test period demand
    demand_vol = []
    for uid, log in panel._logs.items():
        if uid not in user_ids:
            continue
        split = int(log.num_records * TRAIN_FRACTION)
        test_spend = np.sum(log.cost_vectors[split:] * log.action_vectors[split:], axis=1)
        demand_vol.append(float(np.std(test_spend)))
    demand_vol = np.array(demand_vol)

    targets_dict = {
        "High Demand": (high_demand, "classification"),
        "Demand Change": (spend_change, "regression"),
    }

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, max_users=None) -> list[BenchmarkResult]:
    """Run all M5 benchmarks."""
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, max_users)

    if X_rp is None:
        return []

    results = []
    for target_name, (y, task_type) in targets_dict.items():
        print(f"  [{DATASET_NAME}] Target: {target_name} ({task_type})")
        if task_type == "classification":
            pos_rate = np.mean(y)
            if pos_rate < 0.02 or pos_rate > 0.98:
                print(f"    Skipping - too imbalanced (pos_rate={pos_rate:.3f})")
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
