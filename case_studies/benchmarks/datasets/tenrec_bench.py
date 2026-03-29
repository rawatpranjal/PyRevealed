"""Tenrec benchmark: high engagement, loyalty, and novelty prediction.

5M users, 140M interactions; menu-based (recommendation clicks → likes).
Source: https://github.com/yuangh-x/2022-NIPS-Tenrec
"""

from __future__ import annotations

import time as _time
import tracemalloc

import numpy as np

from prefgraph.core.session import MenuChoiceLog

from case_studies.benchmarks.config import TRAIN_FRACTION, MIN_OBS_MENU, MIN_TRAIN_MENU, MIN_TEST_MENU
from case_studies.benchmarks.core.features import extract_menu_baseline, extract_menu_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "Tenrec"


def _split_menu_log(log: MenuChoiceLog, fraction: float):
    """Split a MenuChoiceLog temporally, remapping items in each half."""
    T = len(log.choices)
    split = int(T * fraction)

    def _remap(menus, choices):
        all_items: set[int] = set()
        for m in menus:
            all_items |= set(m)
        item_map = {item: idx for idx, item in enumerate(sorted(all_items))}
        return MenuChoiceLog(
            menus=[frozenset(item_map[i] for i in m) for m in menus],
            choices=[item_map[c] for c in choices],
        )

    return _remap(log.menus[:split], log.choices[:split]), _remap(log.menus[split:], log.choices[split:])


def load_and_prepare(data_dir=None, max_users=50000):
    """Load Tenrec and prepare train/target splits.

    Targets (computed on test window only):
      - High Engagement: top tercile of test session count
      - Low Loyalty: top tercile of choice dispersion (1 - modal concentration)
      - High Novelty: top tercile of novel choice fraction vs train
    """
    from prefgraph.datasets._tenrec import load_tenrec

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    _t_load = _time.perf_counter()
    user_logs = load_tenrec(data_dir=data_dir, min_sessions=MIN_OBS_MENU, max_users=max_users)
    load_and_prepare.load_time_s = _time.perf_counter() - _t_load

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

        # Engagement: number of sessions in test window
        raw_engagement.append(len(test_log.choices))

        # Choice concentration (loyalty proxy): modal-item share in test choices
        if len(test_log.choices) > 0:
            counts: dict[int, int] = {}
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

    from case_studies.benchmarks.core.eda import compute_menu_eda
    load_and_prepare.eda = compute_menu_eda(train_logs)

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

    engagement = np.array(raw_engagement)
    concentration = np.array(raw_concentration)
    novelty = np.array(raw_novelty)

    targets_dict = {
        "High Engagement": (
            (engagement > np.percentile(engagement, 66.67)).astype(int),
            "classification", engagement, 66.67,
        ),
        "Low Loyalty": (
            (concentration < np.percentile(concentration, 33.33)).astype(int),
            "classification", 1.0 - concentration, 66.67,
        ),
        "High Novelty": (
            (novelty > np.percentile(novelty, 66.67)).astype(int),
            "classification", novelty, 66.67,
        ),
    }

    for tname, (y, _, _, _) in targets_dict.items():
        print(f"  Target '{tname}': pos_rate={np.mean(y):.3f}")

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, max_users=50000) -> list[BenchmarkResult]:
    """Run all Tenrec benchmarks with multiple targets."""
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, max_users)

    if X_rp is None:
        return []

    _load_t = getattr(load_and_prepare, "load_time_s", 0.0)
    _engine_t = getattr(load_and_prepare, "engine_time_s", 0.0)
    _feat_t = getattr(load_and_prepare, "feature_time_s", 0.0)
    _mem = getattr(load_and_prepare, "peak_memory_mb", 0.0)

    results: list[BenchmarkResult] = []
    for target_name, (y, task_type, y_cont, pctl) in targets_dict.items():
        print(f"  [{DATASET_NAME}] Target: {target_name} ({task_type})")
        pos_rate = float(np.mean(y))
        if pos_rate < 0.02 or pos_rate > 0.98:
            print(f"    Skipping - too imbalanced (pos_rate={pos_rate:.3f})")
            continue

        result = run_three_way(
            X_rp, X_base, y, DATASET_NAME, target_name, task_type,
            y_continuous=np.asarray(y_cont), threshold_pctl=pctl,
        )
        result.load_time_s = _load_t
        result.engine_time_s = _engine_t
        result.feature_time_s = _feat_t
        result.peak_memory_mb = _mem
        results.append(result)

        print(
            f"    AUC: RP={result.auc_rp:.3f}  Base={result.auc_base:.3f}  "
            f"Combined={result.auc_combined:.3f}  Lift={result.auc_combined - result.auc_base:+.3f}"
        )

    return results
