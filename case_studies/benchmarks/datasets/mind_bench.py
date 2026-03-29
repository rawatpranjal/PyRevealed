"""MIND news recommendation benchmark: engagement, loyalty, novelty, and CTR prediction.

MIND (Wu et al., ACL 2020) contains 1M+ impression logs from Microsoft News.
Each impression is a natural menu-choice observation: articles shown = menu,
clicked article = choice. MIND-small has ~50K users, ~230K impressions in train.

Targets (computed on test window: last 30% of impressions per user):
  - High Engagement: click count in test window, top tercile.
                     Measures which users are the heaviest news consumers.
  - Low Loyalty:     Shannon entropy of chosen article IDs in test window, top tercile.
                     High entropy = diverse reader (low loyalty to topics/sources).
                     Uses news.tsv category labels when available; falls back to
                     article-ID entropy as a proxy.
  - High Novelty:    fraction of test choices NOT seen in train choices, top tercile.
                     Measures tendency to explore new articles vs repeat reads.
  - High CTR:        inverse of mean menu size in test window (1 / mean_menu_size),
                     top tercile. Since each observation is a 1-click impression,
                     users with smaller menus have higher effective CTR (1 click out
                     of fewer candidates). This proxies ad-click-through on news
                     platforms where smaller, more targeted slates drive higher CTR.
"""

from __future__ import annotations

import numpy as np
from prefgraph.core.session import MenuChoiceLog

from case_studies.benchmarks.config import TRAIN_FRACTION, MIN_OBS_MENU, MIN_TRAIN_MENU, MIN_TEST_MENU
from case_studies.benchmarks.core.features import extract_menu_baseline, extract_menu_rp
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "MIND"


def _split_menu_log(log: MenuChoiceLog, fraction: float):
    """Split a MenuChoiceLog temporally into train and test halves.

    Remaps item IDs independently in each half so both halves have compact
    0..N-1 integer IDs required by the Rust Engine.
    """
    T = len(log.choices)
    split = int(T * fraction)

    def _remap(menus, choices):
        all_items: set[int] = set()
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
    """Load MIND and prepare train/target splits and targets.

    Targets (computed on test window only):
      - High Engagement: number of 1-click impressions in test window
      - Low Loyalty:     Shannon entropy of clicked article IDs in test window
                         (high entropy = reads many distinct articles = low loyalty)
      - High Novelty:    fraction of unique test choices not in train choices
    """
    import time as _time
    import tracemalloc
    from math import log2

    from prefgraph.datasets._mind import load_mind

    print(f"\n[{DATASET_NAME}] Loading dataset...")
    _t_load = _time.perf_counter()
    user_logs = load_mind(
        data_dir=data_dir,
        split="train",
        min_sessions=MIN_OBS_MENU,
        max_users=max_users,
    )
    load_and_prepare.load_time_s = _time.perf_counter() - _t_load

    train_logs: dict[str, MenuChoiceLog] = {}
    user_ids: list[str] = []

    # Raw target primitives — accumulated over the test window
    raw_engagement: list[int] = []       # click count in test window
    raw_loyalty_entropy: list[float] = []  # Shannon entropy of article choices in test
    raw_novelty: list[float] = []         # fraction of new choices in test vs train
    raw_ctr: list[float] = []            # 1 / mean_menu_size in test (effective CTR)

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

        # --- Target 1: High Engagement ---
        # Number of qualifying impressions in the test window.
        # MIND impressions are roughly chronological within a user's block.
        raw_engagement.append(len(test_log.choices))

        # --- Target 2: Low Loyalty (Shannon entropy of article choices) ---
        # Higher entropy = user clicks many different articles = low loyalty.
        # Uses remapped integer article IDs (available from the log).
        if len(test_log.choices) > 0:
            counts: dict[int, int] = {}
            for c in test_log.choices:
                counts[c] = counts.get(c, 0) + 1
            total = sum(counts.values())
            entropy = -sum(
                (cnt / total) * log2(cnt / total)
                for cnt in counts.values()
                if cnt > 0
            )
            raw_loyalty_entropy.append(entropy)
        else:
            raw_loyalty_entropy.append(0.0)

        # --- Target 3: High Novelty ---
        # Fraction of unique test choices not seen in the train choices.
        # High novelty => user explores; low novelty => user repeats.
        train_items = set(train_log.choices)
        test_items = set(test_log.choices)
        if len(test_items) > 0:
            novelty = len(test_items - train_items) / len(test_items)
        else:
            novelty = 0.0
        raw_novelty.append(novelty)

        # --- Target 4: High CTR ---
        # Each MIND observation is a 1-click impression, so CTR per impression
        # is 1 / menu_size. The user-level effective CTR is the mean across
        # test-window impressions: mean(1 / menu_size_t). Users who click from
        # smaller, more targeted slates have higher effective CTR — this proxies
        # ad-click-through rate on news platforms.
        if len(test_log.menus) > 0:
            ctr = float(np.mean([1.0 / len(m) for m in test_log.menus]))
        else:
            ctr = 0.0
        raw_ctr.append(ctr)

    print(f"  Users: {len(user_ids)}")

    if len(user_ids) < 30:
        print(f"  Too few users, skipping.")
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

    engagement = np.array(raw_engagement)
    loyalty_entropy = np.array(raw_loyalty_entropy)
    novelty = np.array(raw_novelty)
    ctr = np.array(raw_ctr)

    # Convert to 4-tuple targets: (y_binary, task_type, y_continuous, threshold_pctl)
    # Top tercile = top 33.3% → threshold at 66.67th percentile
    targets_dict = {
        # High Engagement: top tercile of test-window click count
        "High Engagement": (
            (engagement > np.percentile(engagement, 66.67)).astype(int),
            "classification", engagement, 66.67,
        ),
        # Low Loyalty: top tercile of entropy (= most diverse readers)
        "Low Loyalty": (
            (loyalty_entropy > np.percentile(loyalty_entropy, 66.67)).astype(int),
            "classification", loyalty_entropy, 66.67,
        ),
        # High Novelty: top tercile of novel choice fraction
        "High Novelty": (
            (novelty > np.percentile(novelty, 66.67)).astype(int),
            "classification", novelty, 66.67,
        ),
        # High CTR: top tercile of effective click-through rate (1/menu_size)
        "High CTR": (
            (ctr > np.percentile(ctr, 66.67)).astype(int),
            "classification", ctr, 66.67,
        ),
    }

    for tname, (y, _, _, _) in targets_dict.items():
        print(f"  Target '{tname}': pos_rate={np.mean(y):.3f}")

    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(data_dir=None, max_users=50000) -> list[BenchmarkResult]:
    """Run all MIND benchmarks with multiple targets."""
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
