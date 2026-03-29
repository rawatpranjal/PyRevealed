"""Taobao User Behavior benchmark: high engagement, preference drift.

988K users, 100M events, menu-based (views → purchases).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from prefgraph.core.session import MenuChoiceLog

from case_studies.benchmarks.config import TRAIN_FRACTION, MIN_OBS_MENU, MIN_TRAIN_MENU, MIN_TEST_MENU
from case_studies.benchmarks.core.features import extract_menu_baseline, extract_menu_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult

DATASET_NAME = "Taobao"


def _split_menu_log(log: MenuChoiceLog, fraction: float):
    T = len(log.choices)
    split = int(T * fraction)

    def _remap(menus, choices):
        all_items = set()
        for m in menus:
            all_items |= set(m)
        item_map = {item: idx for idx, item in enumerate(sorted(all_items))}
        return MenuChoiceLog(
            menus=[frozenset(item_map[i] for i in m) for m in menus],
            choices=[item_map[c] for c in choices],
        )

    return _remap(log.menus[:split], log.choices[:split]), _remap(log.menus[split:], log.choices[split:])


def load_and_prepare(data_dir=None, max_users=50000):
    import time as _time
    import tracemalloc

    from prefgraph.datasets._taobao import load_taobao

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    _t_load = _time.perf_counter()
    user_logs = load_taobao(data_dir=data_dir, min_sessions=MIN_OBS_MENU, max_users=max_users)
    load_and_prepare.load_time_s = _time.perf_counter() - _t_load

    train_logs, user_ids = {}, []
    targets = {"high_engagement": []}

    for uid, log in user_logs.items():
        T = len(log.choices)
        if T < MIN_OBS_MENU:
            continue
        split = int(T * TRAIN_FRACTION)
        if split < MIN_TRAIN_MENU or (T - split) < MIN_TEST_MENU:
            continue

        train_log, test_log = _split_menu_log(log, TRAIN_FRACTION)
        train_logs[uid] = train_log
        user_ids.append(uid)
        targets["high_engagement"].append(len(test_log.choices))

    engagement = np.array(targets["high_engagement"])
    threshold = np.percentile(engagement, 66.67)
    high_eng = (engagement > threshold).astype(int)

    print(f"  Users: {len(user_ids)}")
    if len(user_ids) < 30:
        return None, None, {}, user_ids

    print(f"  Extracting baseline features...")
    X_base = extract_menu_baseline(train_logs)

    print(f"  Extracting RP features via Engine...")
    tracemalloc.start()
    _t_feat = _time.perf_counter()
    X_rp = extract_menu_rp(train_logs)
    load_and_prepare.feature_time_s = _time.perf_counter() - _t_feat
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    load_and_prepare.peak_memory_mb = peak_mem / (1024 * 1024)
    load_and_prepare.engine_time_s = getattr(extract_menu_rp, "engine_time_s", 0.0)

    print(f"  Engine scoring: {load_and_prepare.engine_time_s:.1f}s  "
          f"Feature extraction: {load_and_prepare.feature_time_s:.1f}s  "
          f"Peak memory: {load_and_prepare.peak_memory_mb:.0f} MB")

    targets_dict = {"High Engagement": (high_eng, "classification", engagement, 66.67)}
    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, max_users=50000) -> list[BenchmarkResult]:
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, max_users)
    if X_rp is None:
        return []

    _load_t = getattr(load_and_prepare, "load_time_s", 0.0)
    _engine_t = getattr(load_and_prepare, "engine_time_s", 0.0)
    _feat_t = getattr(load_and_prepare, "feature_time_s", 0.0)
    _mem = getattr(load_and_prepare, "peak_memory_mb", 0.0)

    results = []
    for target_name, (y, task_type, y_cont, pctl) in targets_dict.items():
        print(f"  [{DATASET_NAME}] Target: {target_name} ({task_type})")
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
        print(f"    AUC: RP={result.auc_rp:.3f}  Base={result.auc_base:.3f}  "
              f"Combined={result.auc_combined:.3f}  Lift={result.auc_combined - result.auc_base:+.3f}")
    return results
