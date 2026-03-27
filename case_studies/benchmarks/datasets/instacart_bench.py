"""Instacart Market Basket benchmark: reorder, churn, basket size prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import TRAIN_FRACTION, MIN_OBS_BUDGET, MIN_TRAIN_BUDGET, MIN_TEST_BUDGET
from case_studies.benchmarks.core.features import extract_budget_baseline, extract_budget_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "Instacart"

# Instacart has 21 departments — we aggregate quantities at department level.
# Prices are not available so we use uniform prices (quantity-only consistency).
NUM_DEPARTMENTS = 21


def load_and_prepare(data_dir=None, max_users=50000):
    """Load Instacart and prepare train/target splits.

    Since Instacart doesn't have price data, we use uniform prices ($1 per unit).
    This reduces GARP to checking whether the consumer buys strictly dominated
    bundles — a well-precedented approach when prices are unavailable.

    Args:
        data_dir: Path to Instacart data directory containing orders.csv,
            order_products__prior.csv, products.csv, departments.csv.
        max_users: Cap on number of users (default: 5000 for speed).
    """
    from pyrevealed.datasets._instacart import load_instacart

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    panel = load_instacart(data_dir=data_dir, max_users=max_users, min_orders=MIN_OBS_BUDGET)

    user_ids = []
    train_tuples = []
    targets = {"spend_drop": [], "test_total_qty": [], "basket_size": []}

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

        # Targets from test window
        train_mean_qty = float(np.mean(np.sum(qty_train, axis=1)))
        test_mean_qty = float(np.mean(np.sum(qty_test, axis=1)))
        test_total_qty = np.sum(qty_test)

        # Spend drop: did mean basket size drop by >30%?
        qty_ratio = test_mean_qty / max(train_mean_qty, 1e-6)
        targets["spend_drop"].append(1 if qty_ratio < 0.7 else 0)

        # High value: top tercile (computed after loop)
        targets["test_total_qty"].append(float(test_total_qty))

        # Basket size: mean items per order in test window (regression)
        targets["basket_size"].append(test_mean_qty)

    print(f"  Users: {len(user_ids)}")

    print(f"  Extracting baseline features...")
    X_base = extract_budget_baseline(train_tuples, user_ids)

    print(f"  Extracting RP features via Engine...")
    X_rp = extract_budget_rp(train_tuples, user_ids)

    # High value: top tercile of test-window total quantity
    test_qtys = np.array(targets["test_total_qty"])
    threshold = np.percentile(test_qtys, 66.67)
    high_value = (test_qtys > threshold).astype(int)

    targets_dict = {
        "Spend Drop": (np.array(targets["spend_drop"]), "classification"),
        "High Value": (high_value, "classification"),
        "Basket Size": (np.array(targets["basket_size"]), "regression"),
    }

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, max_users=50000) -> list[BenchmarkResult]:
    """Run all Instacart benchmarks."""
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, max_users)

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
