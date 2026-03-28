"""Yoochoose RecSys 2015 benchmark: purchase prediction, repeat visitor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prefgraph.core.session import MenuChoiceLog

from case_studies.benchmarks.config import TRAIN_FRACTION, MIN_OBS_MENU
from case_studies.benchmarks.core.features import extract_menu_baseline, extract_menu_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "Yoochoose"


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


def load_and_prepare(data_dir=None, max_users=5000):
    """Load Yoochoose and prepare train/target splits."""
    from prefgraph.datasets._yoochoose import load_yoochoose

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    user_logs = load_yoochoose(
        data_dir=data_dir,
        min_sessions=MIN_OBS_MENU,
        max_users=max_users,
    )

    train_logs = {}
    targets = {"repeat_visitor": []}
    user_ids = []

    for uid, log in user_logs.items():
        T = len(log.choices)
        if T < MIN_OBS_MENU:
            continue

        split = int(T * TRAIN_FRACTION)
        if split < 3 or (T - split) < 2:
            continue

        train_log, test_log = _split_menu_log(log, TRAIN_FRACTION)
        train_logs[uid] = train_log
        user_ids.append(uid)

        # Repeat visitor: has sessions in test window
        targets["repeat_visitor"].append(1 if len(test_log.choices) > 0 else 0)

    print(f"  Users: {len(user_ids)}")

    print(f"  Extracting baseline features...")
    X_base = extract_menu_baseline(train_logs)

    print(f"  Extracting RP features via Engine...")
    X_rp = extract_menu_rp(train_logs)

    targets_dict = {
        "Repeat Visitor": (np.array(targets["repeat_visitor"]), "classification"),
    }

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, max_users=5000) -> list[BenchmarkResult]:
    """Run all Yoochoose benchmarks."""
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, max_users)

    results = []
    for target_name, (y, task_type) in targets_dict.items():
        print(f"  [{DATASET_NAME}] Target: {target_name} ({task_type})")
        pos_rate = np.mean(y)
        if pos_rate < 0.02 or pos_rate > 0.98:
            print(f"    Skipping - too imbalanced (pos_rate={pos_rate:.3f})")
            continue

        result = run_three_way(X_rp, X_base, y, DATASET_NAME, target_name, task_type)
        results.append(result)
        print(f"    AUC: RP={result.auc_rp:.3f}  Base={result.auc_base:.3f}  "
              f"Combined={result.auc_combined:.3f}  Lift={result.auc_combined - result.auc_base:+.3f}")

    return results
