"""FINN.no Recsys Slates benchmark: engagement, loyalty, and search-intent prediction.

Observed-slate dataset from Norway's largest classifieds marketplace.
Unlike session-reconstructed menus (Taobao, REES46), the choice set here is
directly logged by the platform — the strongest WARP/SARP evidence.

Source: Eide et al. (2021). RecSys 2021.
Dataset: https://github.com/finn-no/recsys_slates_dataset

Targets (computed on the test window: last 30% of interactions per user):
  - High Engagement:   top tercile of test session count.
  - Low Loyalty:       top tercile of choice dispersion (1 - modal concentration).
  - High Search Ratio: top tercile of fraction of clicks originating from search
                       (interaction_type == 1) rather than recommendations
                       (interaction_type == 2). High search fraction signals
                       active purchase intent — the user is looking for something
                       specific rather than passively browsing the feed. This is
                       the closest revenue-intent proxy in the dataset; FINN.no
                       item IDs are unique per listing so novelty is trivially 1.0.
"""

from __future__ import annotations

import time as _time
import tracemalloc

import numpy as np

from prefgraph.core.session import MenuChoiceLog

from case_studies.benchmarks.config import TRAIN_FRACTION, MIN_OBS_MENU, MIN_TRAIN_MENU, MIN_TEST_MENU
from case_studies.benchmarks.core.features import extract_menu_baseline, extract_menu_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "FINN.no Slates"


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


def _load_search_ratios(
    data_dir,
    max_users: int | None,
    qualifying_user_ids: set[str],
) -> dict[str, list[int]]:
    """Return per-user sequence of interaction types for valid clicks.

    Loads only the 'click' and 'interaction_type' arrays from data.npz (not the
    3GB 'slate' array). For each qualifying user row i, records the
    interaction_type for every timestep where click[i, t] >= PADDING_THRESHOLD,
    in the same temporal (timestep) order as the MenuChoiceLog.

    interaction_type values: 1 = user-initiated search, 2 = recommendation.

    Args:
        data_dir: Same data_dir argument passed to load_finn_slates.
        max_users: Same row cap applied in load_finn_slates.
        qualifying_user_ids: Set of uid strings (row indices) from user_logs.

    Returns:
        Dict mapping uid_str → list of int interaction_type values for each
        valid click, in chronological order.
    """
    from prefgraph.datasets._finn_slates import (
        _find_data_dir as _find_finn_dir,
        PADDING_THRESHOLD,
    )

    data_path = _find_finn_dir(data_dir)
    npz_path = data_path / "data.npz"

    qualifying_indices = {int(uid) for uid in qualifying_user_ids}

    # Load only the two small arrays — avoids decompressing the 3GB slate array.
    with np.load(npz_path, allow_pickle=False) as d:
        click = d["click"]                    # [N, T] int32
        interaction_type = d["interaction_type"]  # [N, T] int32

    # Apply the same user-count cap as the loader (row-slice).
    if max_users is not None:
        click = click[:max_users]
        interaction_type = interaction_type[:max_users]

    result: dict[str, list[int]] = {}
    T = click.shape[1]
    for i in qualifying_indices:
        if i >= len(click):
            continue
        types = [
            int(interaction_type[i, t])
            for t in range(T)
            if int(click[i, t]) >= PADDING_THRESHOLD
        ]
        if types:
            result[str(i)] = types

    return result


def load_and_prepare(data_dir=None, max_users=100_000):
    """Load FINN.no Slates and prepare train/target splits.

    Targets (computed on test window only):
      - High Engagement:   top tercile of test session count.
      - Low Loyalty:       top tercile of choice dispersion (1 - modal concentration).
      - High Search Ratio: top tercile of fraction of clicked interactions that
                           originated from search (interaction_type == 1) vs
                           recommendations (interaction_type == 2) in the test window.
                           High search ratio proxies active purchase intent on a
                           classifieds marketplace. "High Novelty" is not used here
                           because every FINN.no listing has a unique item ID, making
                           the novelty fraction trivially 1.0 for all users.
    """
    from prefgraph.datasets._finn_slates import load_finn_slates

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    _t_load = _time.perf_counter()
    user_logs = load_finn_slates(
        data_dir=data_dir,
        min_sessions=MIN_OBS_MENU,
        max_users=max_users,
    )
    load_and_prepare.load_time_s = _time.perf_counter() - _t_load

    # Load interaction types for the High Search Ratio target.
    # Only click and interaction_type arrays are read (not the heavy slate array).
    print(f"  Loading interaction types for search-ratio target...")
    user_it_map = _load_search_ratios(data_dir, max_users, set(user_logs.keys()))

    train_logs: dict[str, MenuChoiceLog] = {}
    user_ids: list[str] = []
    raw_engagement: list[int] = []
    raw_concentration: list[float] = []
    raw_search_ratio: list[float] = []

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

        # Target 1: High Engagement — test session count
        raw_engagement.append(len(test_log.choices))

        # Target 2: Low Loyalty — modal-item share in test choices
        if len(test_log.choices) > 0:
            counts: dict[int, int] = {}
            for c in test_log.choices:
                counts[c] = counts.get(c, 0) + 1
            modal = max(counts.values())
            raw_concentration.append(modal / len(test_log.choices))
        else:
            raw_concentration.append(0.0)

        # Target 3: High Search Ratio — fraction of test-window clicks from search.
        # The interaction_type sequence aligns with the MenuChoiceLog: one entry
        # per valid click (click >= PADDING_THRESHOLD), same temporal order.
        it_seq = user_it_map.get(uid, [])
        if len(it_seq) >= T:
            test_it = it_seq[split:T]
            if test_it:
                search_frac = sum(1 for x in test_it if x == 1) / len(test_it)
            else:
                search_frac = 0.0
        else:
            search_frac = 0.0
        raw_search_ratio.append(search_frac)

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
    search_ratio = np.array(raw_search_ratio)

    targets_dict = {
        "High Engagement": (
            (engagement > np.percentile(engagement, 66.67)).astype(int),
            "classification", engagement, 66.67,
        ),
        "Low Loyalty": (
            (concentration < np.percentile(concentration, 33.33)).astype(int),
            "classification", 1.0 - concentration, 66.67,
        ),
        "High Search Ratio": (
            (search_ratio > np.percentile(search_ratio, 66.67)).astype(int),
            "classification", search_ratio, 66.67,
        ),
    }

    for tname, (y, _, _, _) in targets_dict.items():
        print(f"  Target '{tname}': pos_rate={np.mean(y):.3f}")

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, max_users=100_000) -> list[BenchmarkResult]:
    """Run all FINN.no Slates benchmarks with multiple targets."""
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
