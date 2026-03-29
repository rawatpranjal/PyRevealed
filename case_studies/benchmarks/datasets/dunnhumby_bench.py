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

    After execution, ``load_and_prepare.load_time_s``, ``.engine_time_s``,
    ``.feature_time_s``, and ``.peak_memory_mb`` hold performance metrics.
    """
    import time as _time
    import tracemalloc

    from prefgraph.datasets import load_dunnhumby

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    _t_load = _time.perf_counter()
    panel = load_dunnhumby(data_dir=data_dir, n_households=n_households, min_weeks=MIN_OBS_BUDGET)
    load_and_prepare.load_time_s = _time.perf_counter() - _t_load

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
    tracemalloc.start()
    _t_feat = _time.perf_counter()
    X_rp_engine = extract_budget_rp(train_tuples, user_ids)
    load_and_prepare.feature_time_s = _time.perf_counter() - _t_feat
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    load_and_prepare.peak_memory_mb = peak_mem / (1024 * 1024)
    load_and_prepare.engine_time_s = getattr(extract_budget_rp, "engine_time_s", 0.0)

    print(f"  Engine scoring: {load_and_prepare.engine_time_s:.1f}s  "
          f"Feature extraction: {load_and_prepare.feature_time_s:.1f}s  "
          f"Peak memory: {load_and_prepare.peak_memory_mb:.0f} MB")

    X_rp = X_rp_engine

    # --- Targets ---

    # Churn: mean spend dropped by >50% from train to test window
    spend_ratio = test_mean_spends / np.maximum(train_mean_spends, 1e-6)
    churn = (spend_ratio < 0.5).astype(int)

    # High spender: top tercile of test-window total spend
    threshold = np.percentile(test_total_spends, 66.67)
    high_spender = (test_total_spends > threshold).astype(int)

    # targets_dict: name -> (y, task_type, y_continuous, threshold_pctl)
    # y_continuous + threshold_pctl = let evaluation.py binarize on train only (zero leakage)
    # Spend Change dropped: R² negative for all models (target is unpredictable)
    targets_dict = {
        "Spend Drop": (churn, "classification", None, None),
        "High Spender": (high_spender, "classification", test_total_spends, 66.67),
        "Future LTV": (test_mean_spends, "regression", None, None),
    }

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, n_households=None) -> list[BenchmarkResult]:
    """Run all Dunnhumby benchmarks."""
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, n_households)

    # Capture pipeline timing from load_and_prepare
    _load_t = getattr(load_and_prepare, "load_time_s", 0.0)
    _engine_t = getattr(load_and_prepare, "engine_time_s", 0.0)
    _feat_t = getattr(load_and_prepare, "feature_time_s", 0.0)
    _mem = getattr(load_and_prepare, "peak_memory_mb", 0.0)

    results = []
    for target_name, (y, task_type, y_cont, pctl) in targets_dict.items():
        print(f"  [{DATASET_NAME}] Target: {target_name} ({task_type})")
        if task_type == "classification":
            pos_rate = np.mean(y)
            if pos_rate < 0.02 or pos_rate > 0.98:
                print(f"    Skipping - too imbalanced (pos_rate={pos_rate:.3f})")
                continue

        result = run_three_way(X_rp, X_base, y, DATASET_NAME, target_name, task_type,
                               y_continuous=y_cont, threshold_pctl=pctl)
        result.load_time_s = _load_t
        result.engine_time_s = _engine_t
        result.feature_time_s = _feat_t
        result.peak_memory_mb = _mem
        results.append(result)

        if task_type == "classification":
            print(f"    AUC: RP={result.auc_rp:.3f}  Base={result.auc_base:.3f}  "
                  f"Combined={result.auc_combined:.3f}  Lift={result.auc_combined - result.auc_base:+.3f}")
        else:
            print(f"    R2: RP={result.r2_rp:.3f}  Base={result.r2_base:.3f}  "
                  f"Combined={result.r2_combined:.3f}")

    return results
