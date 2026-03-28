"""REES46 eCommerce benchmark: purchase prediction, preference drift."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prefgraph.core.session import MenuChoiceLog

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
    """Load REES46 and prepare train/target splits and targets.

    Targets (computed on test window only):
      - High Engagement: top tercile of test session count
      - Low Loyalty: bottom tercile of modal-choice concentration in test
      - High Novelty: top tercile of novel choice fraction in test (vs train)
    """
    from prefgraph.datasets._rees46 import load_rees46

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    user_logs = load_rees46(
        data_dir=data_dir,
        min_sessions=MIN_OBS_MENU,
        max_users=max_users,
    )

    train_logs: dict[str, MenuChoiceLog] = {}
    user_ids: list[str] = []
    raw_engagement: list[int] = []
    raw_concentration: list[float] = []
    raw_novelty: list[float] = []

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

        # --- Target primitives (test window only) ---
        # Engagement: number of sessions in test window
        raw_engagement.append(len(test_log.choices))

        # Choice concentration: share of the modal item in test choices
        if len(test_log.choices) > 0:
            counts = {}
            for c in test_log.choices:
                counts[c] = counts.get(c, 0) + 1
            modal = max(counts.values())
            raw_concentration.append(modal / len(test_log.choices))
        else:
            raw_concentration.append(0.0)

        # Novelty: fraction of unique test choices not seen in train
        train_items = set(train_log.choices)
        test_items = set(test_log.choices)
        if len(test_items) > 0:
            raw_novelty.append(len(test_items - train_items) / len(test_items))
        else:
            raw_novelty.append(0.0)

    print(f"  Users: {len(user_ids)}")

    if len(user_ids) < 30:
        print(f"  Too few users, skipping.")
        return None, None, {}, user_ids

    print(f"  Extracting baseline features...")
    X_base = extract_menu_baseline(train_logs)

    print(f"  Extracting RP features via Engine...")
    X_rp = extract_menu_rp(train_logs)

    # Convert raw target primitives to dictionary with leakage-safe thresholds
    engagement = np.array(raw_engagement)
    concentration = np.array(raw_concentration)
    novelty = np.array(raw_novelty)

    targets_dict = {
        # 1) High Engagement: top tercile of test sessions
        "High Engagement": (
            (engagement > np.percentile(engagement, 66.67)).astype(int),
            "classification", engagement, 66.67,
        ),
        # 2) Low Loyalty: top tercile of dispersion = 1 - concentration
        "Low Loyalty": (
            (concentration < np.percentile(concentration, 33.33)).astype(int),
            "classification", 1.0 - concentration, 66.67,
        ),
        # 3) High Novelty: top tercile of novel choice fraction
        "High Novelty": (
            (novelty > np.percentile(novelty, 66.67)).astype(int),
            "classification", novelty, 66.67,
        ),
    }

    for tname, (y, _, _, _) in targets_dict.items():
        print(f"  Target '{tname}': pos_rate={np.mean(y):.3f}")

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, max_users=50000) -> list[BenchmarkResult]:
    """Run all REES46 benchmarks with multiple targets."""
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, max_users)

    if X_rp is None:
        return []

    results: list[BenchmarkResult] = []
    for target_name, target_tuple in targets_dict.items():
        # Support leakage-safe binarization using train-only threshold
        if len(target_tuple) == 4:
            y, task_type, y_cont, pctl = target_tuple
            y_cont = np.asarray(y_cont)
            print(f"  [{DATASET_NAME}] Target: {target_name} ({task_type})")
            pos_rate = float(np.mean(y))
            if pos_rate < 0.02 or pos_rate > 0.98:
                print(f"    Skipping - too imbalanced (pos_rate={pos_rate:.3f})")
                continue
            result = run_three_way(
                X_rp, X_base, y, DATASET_NAME, target_name, task_type,
                y_continuous=y_cont, threshold_pctl=pctl,
            )
        else:
            y, task_type = target_tuple  # Backward compatibility
            print(f"  [{DATASET_NAME}] Target: {target_name} ({task_type})")
            pos_rate = float(np.mean(y))
            if pos_rate < 0.02 or pos_rate > 0.98:
                print(f"    Skipping - too imbalanced (pos_rate={pos_rate:.3f})")
                continue
            result = run_three_way(X_rp, X_base, y, DATASET_NAME, target_name, task_type)

        results.append(result)
        if result.task_type == "classification":
            print(
                f"    AUC: RP={result.auc_rp:.3f}  Base={result.auc_base:.3f}  "
                f"Combined={result.auc_combined:.3f}  Lift={result.auc_combined - result.auc_base:+.3f}"
            )
        else:
            print(
                f"    R2:  RP={result.r2_rp:.3f}  Base={result.r2_base:.3f}  "
                f"Combined={result.r2_combined:.3f}  Δ={result.r2_combined - result.r2_base:+.3f}"
            )

    return results
