"""H&M Fashion benchmark: high spender, spend change, future spend.

1.36M customers, 31.8M transactions (2018-2020). Budget-based with real prices.
Aggregated to monthly periods by product group.

Price construction: per-customer realized prices for purchased groups,
period-group median imputation for unpurchased groups.

Targets:
  - High Spender: top tercile of target-window total spend (classification)
  - Spend Change: target mean spend minus train mean spend (regression)
  - Future Spend: mean spend per period in target window (regression)

Runs dual baselines (core 10-feature vs full 17-feature) to test
whether RP features add marginal predictive value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from case_studies.benchmarks.config import TRAIN_FRACTION, MIN_OBS_BUDGET, MIN_TRAIN_BUDGET, MIN_TEST_BUDGET
from case_studies.benchmarks.core.features import extract_budget_baseline, extract_budget_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "H&M"


def _print_eda(train_tuples, user_ids):
    """Print panel-level EDA summary after loading."""
    n_users = len(user_ids)
    obs_counts = [t[0].shape[0] for t in train_tuples]

    all_spends = []
    realized_prices = []
    group_counts = []
    for prices, qty in train_tuples:
        spend_per_obs = np.sum(prices * qty, axis=1)
        all_spends.extend(spend_per_obs.tolist())
        mask = qty > 0
        if mask.any():
            realized_prices.extend(prices[mask].tolist())
        active = np.sum(qty > 0, axis=1)
        group_counts.extend(active.tolist())

    all_spends = np.array(all_spends)
    realized_prices = np.array(realized_prices)
    group_counts = np.array(group_counts)

    hhi_values = []
    for prices, qty in train_tuples:
        total_per_cat = np.sum(qty, axis=0)
        total = np.sum(total_per_cat)
        if total > 0:
            shares = total_per_cat / total
            hhi_values.append(float(np.sum(shares ** 2)))
    hhi_values = np.array(hhi_values)

    print(f"\n  === H&M EDA Summary ===")
    print(f"  Users: {n_users}")
    print(f"  Obs per user: median={np.median(obs_counts):.0f}  "
          f"mean={np.mean(obs_counts):.1f}  min={min(obs_counts)}  max={max(obs_counts)}")
    print(f"  Spend/period: median={np.median(all_spends):.4f}  "
          f"mean={np.mean(all_spends):.4f}  std={np.std(all_spends):.4f}")
    print(f"  Realized prices: median={np.median(realized_prices):.4f}  "
          f"mean={np.mean(realized_prices):.4f}  std={np.std(realized_prices):.4f}")
    print(f"  Active groups/period: median={np.median(group_counts):.1f}  "
          f"mean={np.mean(group_counts):.1f}")
    print(f"  HHI concentration: median={np.median(hhi_values):.3f}  "
          f"mean={np.mean(hhi_values):.3f}")
    print(f"  ========================\n")


def load_and_prepare(data_dir=None, max_users=50000):
    """Load H&M with temporal split. Returns dual baselines (core10 + full17)."""
    import time as _time
    import tracemalloc

    from prefgraph.datasets._hm import load_hm

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    _t_load = _time.perf_counter()
    panel = load_hm(data_dir=data_dir, max_users=max_users, min_periods=6)
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

        train_spend = np.sum(prices_train * qty_train, axis=1)
        test_spend = np.sum(prices_test * qty_test, axis=1)
        train_mean_spends.append(float(np.mean(train_spend)))
        test_mean_spends.append(float(np.mean(test_spend)))
        test_total_spends.append(float(np.sum(test_spend)))

    train_mean_spends = np.array(train_mean_spends)
    test_mean_spends = np.array(test_mean_spends)
    test_total_spends = np.array(test_total_spends)

    print(f"  Users after filtering: {len(user_ids)}")

    from case_studies.benchmarks.core.eda import compute_budget_eda
    load_and_prepare.eda = compute_budget_eda(train_tuples, user_ids)

    if len(user_ids) < 30:
        return None, None, None, {}, user_ids

    # EDA
    _print_eda(train_tuples, user_ids)

    # Dual baselines: core10 has the 10 features from the spec (spend stats +
    # category concentration + trend). full17 adds recency, purchase gaps, and
    # inter-purchase variability. Empirical finding (2026-03-28, 46K users):
    # core10 and full17 produce nearly identical AUC/R2 - the 7 extra features
    # don't carry marginal signal beyond the core 10. Both kept for validation.
    print(f"  Extracting baseline features (core 10)...")
    X_base_10 = extract_budget_baseline(train_tuples, user_ids, feature_set="core")

    print(f"  Extracting baseline features (full 17)...")
    X_base_17 = extract_budget_baseline(train_tuples, user_ids, feature_set="full")

    print(f"  Extracting RP features via Engine...")
    tracemalloc.start()
    _t_feat = _time.perf_counter()
    X_rp = extract_budget_rp(train_tuples, user_ids)
    load_and_prepare.feature_time_s = _time.perf_counter() - _t_feat
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    load_and_prepare.peak_memory_mb = peak_mem / (1024 * 1024)
    load_and_prepare.engine_time_s = getattr(extract_budget_rp, "engine_time_s", 0.0)

    print(f"  Engine scoring: {load_and_prepare.engine_time_s:.1f}s  "
          f"Feature extraction: {load_and_prepare.feature_time_s:.1f}s  "
          f"Peak memory: {load_and_prepare.peak_memory_mb:.0f} MB")

    # Targets
    # High Spender: initial binarization on all users; run_three_way re-binarizes
    # using train-only threshold via y_continuous + threshold_pctl (zero leakage)
    threshold_all = np.percentile(test_total_spends, 66.67)
    high_spender = (test_total_spends > threshold_all).astype(int)
    spend_change = test_mean_spends - train_mean_spends

    # Do NOT call the regression target "LTV" - users with no future window
    # are excluded, so this is future-window spend among survivors, not true
    # lifetime value for the full population. "Churn" was also dropped: the
    # design excludes users who disappear, so "churn" is really "large spend
    # decline among survivors", which is misleading as a headline target.
    targets_dict = {
        "High Spender": (high_spender, "classification", test_total_spends, 66.67),
        "Spend Change": (spend_change, "regression", None, None),
        "Future Spend": (test_mean_spends, "regression", None, None),
    }

    return X_rp, X_base_10, X_base_17, targets_dict, user_ids


def run_benchmark(data_dir=None, max_users=50000) -> list[BenchmarkResult]:
    X_rp, X_base_10, X_base_17, targets_dict, user_ids = load_and_prepare(data_dir, max_users)
    if X_rp is None:
        return []

    _load_t = getattr(load_and_prepare, "load_time_s", 0.0)
    _engine_t = getattr(load_and_prepare, "engine_time_s", 0.0)
    _feat_t = getattr(load_and_prepare, "feature_time_s", 0.0)
    _mem = getattr(load_and_prepare, "peak_memory_mb", 0.0)

    results = []
    for target_name, (y, task_type, y_continuous, threshold_pctl) in targets_dict.items():
        print(f"  [{DATASET_NAME}] Target: {target_name} ({task_type})")

        if task_type == "classification" and y is not None:
            pos_rate = np.mean(y)
            if pos_rate < 0.02 or pos_rate > 0.98:
                print(f"    Skipping - too imbalanced (pos_rate={pos_rate:.3f})")
                continue

        # --- Core 10 baseline ---
        result_10 = run_three_way(
            X_rp, X_base_10, y,
            DATASET_NAME, f"{target_name} (core10)", task_type,
            y_continuous=y_continuous, threshold_pctl=threshold_pctl,
        )
        result_10.load_time_s = _load_t
        result_10.engine_time_s = _engine_t
        result_10.feature_time_s = _feat_t
        result_10.peak_memory_mb = _mem
        results.append(result_10)

        # --- Full 17 baseline ---
        result_17 = run_three_way(
            X_rp, X_base_17, y,
            DATASET_NAME, f"{target_name} (full17)", task_type,
            y_continuous=y_continuous, threshold_pctl=threshold_pctl,
        )
        result_17.load_time_s = _load_t
        result_17.engine_time_s = _engine_t
        result_17.feature_time_s = _feat_t
        result_17.peak_memory_mb = _mem
        results.append(result_17)

        # Print comparison
        if task_type == "classification":
            print(f"    Core10: Base={result_10.auc_base:.3f}  +RP={result_10.auc_combined:.3f}  "
                  f"Lift={result_10.auc_combined - result_10.auc_base:+.3f}")
            print(f"    Full17: Base={result_17.auc_base:.3f}  +RP={result_17.auc_combined:.3f}  "
                  f"Lift={result_17.auc_combined - result_17.auc_base:+.3f}")
        else:
            print(f"    Core10: Base R2={result_10.r2_base:.3f}  +RP R2={result_10.r2_combined:.3f}")
            print(f"    Full17: Base R2={result_17.r2_base:.3f}  +RP R2={result_17.r2_combined:.3f}")

    return results
