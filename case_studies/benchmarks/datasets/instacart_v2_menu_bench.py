"""Instacart V2 menu-choice benchmark: aisle-level single-reorder construction.

Tests whether RP preference-graph features from the V2 construction
(aisle x single-reorder x trailing-3 menu) add predictive lift over
a strong engagement baseline.

Construction:
  Observation = user x order x aisle
  Choice      = sole reordered SKU in that aisle-order
  Menu        = trailing-3 aisle products union {choice}, menu_size >= 2
  Pair filter = (user, aisle) pairs with >= 3 valid events

This replaces instacart_menu_bench.py (V1, broken: dept + first-in-cart).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from prefgraph.core.session import MenuChoiceLog

from case_studies.benchmarks.config import TRAIN_FRACTION, MIN_OBS_MENU, MIN_TRAIN_MENU, MIN_TEST_MENU
from case_studies.benchmarks.core.features import extract_menu_baseline, extract_menu_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult

DATASET_NAME = "Instacart V2 (Menu)"


def _split_menu_log(log: MenuChoiceLog, fraction: float):
    """Temporal train/test split with independent item remapping per half."""
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

    return (
        _remap(log.menus[:split], log.choices[:split]),
        _remap(log.menus[split:], log.choices[split:]),
    )


def print_eda(user_logs: dict[str, MenuChoiceLog]) -> None:
    """Print dataset quality statistics before running the benchmark."""
    n_users = len(user_logs)
    total_sessions = sum(log.num_observations for log in user_logs.values())
    sessions_per_user = [log.num_observations for log in user_logs.values()]
    menu_sizes = [len(m) for log in user_logs.values() for m in log.menus]
    unique_items_per_user = [log.num_items for log in user_logs.values()]

    # Choice concentration: fraction of sessions where user picks their modal item
    choice_concentrations = []
    for log in user_logs.values():
        counts = Counter(log.choices)
        if log.num_observations > 0:
            choice_concentrations.append(counts.most_common(1)[0][1] / log.num_observations)

    # Switching rate: fraction of consecutive sessions where user switches choice
    switch_rates = []
    for log in user_logs.values():
        if log.num_observations > 1:
            switches = sum(
                1 for i in range(1, log.num_observations)
                if log.choices[i] != log.choices[i - 1]
            )
            switch_rates.append(switches / (log.num_observations - 1))

    print(f"\n  {'=' * 60}")
    print(f"  {DATASET_NAME} - EDA Summary")
    print(f"  {'=' * 60}")
    print(f"  Users:              {n_users:,}")
    print(f"  Total sessions:     {total_sessions:,}")
    print(f"  Sessions/user:      median={np.median(sessions_per_user):.0f}  "
          f"p25={np.percentile(sessions_per_user, 25):.0f}  "
          f"p75={np.percentile(sessions_per_user, 75):.0f}  "
          f"max={np.max(sessions_per_user):.0f}")
    print(f"  Menu size:          median={np.median(menu_sizes):.1f}  "
          f"p25={np.percentile(menu_sizes, 25):.1f}  "
          f"p75={np.percentile(menu_sizes, 75):.1f}  "
          f"max={np.max(menu_sizes):.0f}")
    print(f"  Unique items/user:  median={np.median(unique_items_per_user):.0f}  "
          f"max={np.max(unique_items_per_user):.0f}")
    print(f"  Choice concentration (modal item share):")
    print(f"    median={np.median(choice_concentrations):.3f}  "
          f"p75={np.percentile(choice_concentrations, 75):.3f}  "
          f"p90={np.percentile(choice_concentrations, 90):.3f}")
    if switch_rates:
        print(f"  Switch rate (consecutive sessions):")
        print(f"    median={np.median(switch_rates):.3f}  "
              f"p25={np.percentile(switch_rates, 25):.3f}  "
              f"p75={np.percentile(switch_rates, 75):.3f}")
    pct_rich = np.mean(np.array(sessions_per_user) >= 10) * 100
    print(f"  Users with >= 10 sessions: {pct_rich:.1f}%")
    print(f"  {'=' * 60}")


def print_sample_observations(user_logs: dict[str, MenuChoiceLog], n_users: int = 3) -> None:
    """Print a few sample observations to sanity-check the construction."""
    print(f"\n  Sample observations ({n_users} users x first 3 sessions):")
    print(f"  {'─' * 60}")
    sampled_uids = list(user_logs.keys())[:n_users]
    for uid in sampled_uids:
        log = user_logs[uid]
        print(f"  {uid}  ({log.num_observations} sessions, {log.num_items} unique items)")
        for t in range(min(3, log.num_observations)):
            menu = sorted(log.menus[t])
            choice = log.choices[t]
            print(f"    t={t}  menu={menu}  choice={choice}  menu_size={len(menu)}")
    print(f"  {'─' * 60}")


def load_and_prepare(data_dir=None, max_users: int = 50_000):
    """Load Instacart V2 data, split train/test, and extract features."""
    import time as _time
    import tracemalloc

    from prefgraph.datasets._instacart_menu_v2 import load_instacart_menu_v2

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    _t_load = _time.perf_counter()
    user_logs = load_instacart_menu_v2(
        data_dir=data_dir,
        min_sessions=MIN_OBS_MENU,
        max_users=max_users,
    )
    load_and_prepare.load_time_s = _time.perf_counter() - _t_load

    # EDA and sample observations before splitting
    print_eda(user_logs)
    print_sample_observations(user_logs)

    # Temporal split and target construction
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
        raw_engagement.append(len(test_log.choices))

        # Choice Concentration: modal item frequency in test window
        # Low concentration = dispersed choices across many items
        test_counts = Counter(test_log.choices)
        raw_concentration.append(test_counts.most_common(1)[0][1] / len(test_log.choices))

        # Novelty: fraction of unique test choices not seen in train
        # Measures preference exploration vs habit
        train_items = set(train_log.choices)
        test_items = set(test_log.choices)
        if len(test_items) > 0:
            raw_novelty.append(len(test_items - train_items) / len(test_items))
        else:
            raw_novelty.append(0.0)

    engagement = np.array(raw_engagement)
    concentration = np.array(raw_concentration)
    novelty = np.array(raw_novelty)

    print(f"\n  Users after train/test split: {len(user_ids)}")
    if len(user_ids) < 30:
        print("  Too few users - skipping")
        return None, None, {}, user_ids

    # Feature extraction
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

    # --- Targets ---
    # 1. High Engagement: top tercile of test-window session count (standard cross-dataset target)
    # 2. Low Loyalty: top tercile of choice dispersion (1 - concentration)
    #    Whether user spreads test-window choices across many items vs habitual repeat
    # 3. High Novelty: top tercile of novel item fraction in test window
    #    Whether user tries items in test window that weren't chosen in train window
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


def run_benchmark(data_dir=None, max_users: int = 50_000) -> list[BenchmarkResult]:
    """Run the full Instacart V2 menu benchmark and return results."""
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, max_users)
    if X_rp is None:
        return []

    _load_t = getattr(load_and_prepare, "load_time_s", 0.0)
    _engine_t = getattr(load_and_prepare, "engine_time_s", 0.0)
    _feat_t = getattr(load_and_prepare, "feature_time_s", 0.0)
    _mem = getattr(load_and_prepare, "peak_memory_mb", 0.0)

    results: list[BenchmarkResult] = []
    for target_name, (y, task_type, y_cont, pctl) in targets_dict.items():
        print(f"\n  [{DATASET_NAME}] Target: {target_name} ({task_type})")
        pos_rate = float(np.mean(y))
        if pos_rate < 0.02 or pos_rate > 0.98:
            print(f"    Skipping - too imbalanced (pos_rate={pos_rate:.3f})")
            continue

        result = run_three_way(
            X_rp, X_base, y,
            DATASET_NAME, target_name, task_type,
            y_continuous=y_cont, threshold_pctl=pctl,
        )
        result.load_time_s = _load_t
        result.engine_time_s = _engine_t
        result.feature_time_s = _feat_t
        result.peak_memory_mb = _mem
        results.append(result)
        lift = result.auc_combined - result.auc_base
        print(f"    AUC: RP={result.auc_rp:.3f}  Base={result.auc_base:.3f}  "
              f"Combined={result.auc_combined:.3f}  Lift={lift:+.3f}")

    return results
