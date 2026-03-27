"""REES46 eCommerce benchmark: purchase prediction, preference drift."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pyrevealed.core.session import MenuChoiceLog

from case_studies.benchmarks.config import TRAIN_FRACTION, MIN_OBS_MENU, MIN_TRAIN_MENU, MIN_TEST_MENU
from case_studies.benchmarks.core.features import extract_menu_baseline, extract_menu_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "REES46"


def _split_menu_log(log: MenuChoiceLog, fraction: float):
    """Split a MenuChoiceLog temporally, remapping items in each half."""
    T = len(log.choices)
    split = int(T * fraction)

    def _remap(menus, choices):
        all_items = set()
        for m in menus:
            all_items |= set(m)
        item_map = {item: idx for idx, item in enumerate(sorted(all_items))}
        remapped_menus = [frozenset(item_map[i] for i in m) for m in menus]
        remapped_choices = [item_map[c] for c in choices]
        return MenuChoiceLog(menus=remapped_menus, choices=remapped_choices)

    train = _remap(log.menus[:split], log.choices[:split])
    test = _remap(log.menus[split:], log.choices[split:])
    return train, test


def load_and_prepare(data_dir=None, max_users=50000):
    """Load REES46 and prepare train/target splits."""
    from pyrevealed.datasets._rees46 import load_rees46

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    user_logs = load_rees46(
        data_dir=data_dir,
        min_sessions=MIN_OBS_MENU,
        max_users=max_users,
    )

    train_logs = {}
    targets = {"high_engagement": []}
    user_ids = []

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

        # High engagement: above-median number of sessions in test window
        targets["high_engagement"].append(len(test_log.choices))

    # Convert to binary: top tercile (consistent with budget datasets)
    engagement = np.array(targets["high_engagement"])
    threshold = np.percentile(engagement, 66.67)
    high_eng = (engagement > threshold).astype(int)

    print(f"  Users: {len(user_ids)}")

    if len(user_ids) < 30:
        print(f"  Too few users, skipping.")
        return None, None, {}, user_ids

    print(f"  Extracting baseline features...")
    X_base = extract_menu_baseline(train_logs)

    print(f"  Extracting RP features via Engine...")
    X_rp = extract_menu_rp(train_logs)

    targets_dict = {
        "High Engagement": (high_eng, "classification"),
    }

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, max_users=50000) -> list[BenchmarkResult]:
    """Run all REES46 benchmarks."""
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, max_users)

    if X_rp is None:
        return []

    results = []
    for target_name, (y, task_type) in targets_dict.items():
        print(f"  [{DATASET_NAME}] Target: {target_name} ({task_type})")
        pos_rate = np.mean(y)
        if pos_rate < 0.02 or pos_rate > 0.98:
            print(f"    Skipping — too imbalanced (pos_rate={pos_rate:.3f})")
            continue

        result = run_three_way(X_rp, X_base, y, DATASET_NAME, target_name, task_type)
        results.append(result)
        print(f"    AUC: RP={result.auc_rp:.3f}  Base={result.auc_base:.3f}  "
              f"Combined={result.auc_combined:.3f}  Lift={result.auc_lift:+.3f}")

    return results
