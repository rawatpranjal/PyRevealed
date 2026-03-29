"""Taobao (Buy-Anchored Window) menu-choice benchmark.

Construction:
  - Observation is anchored on a buy event
  - Menu = items viewed in the last `window_seconds` before the buy
  - Require bought item was viewed within the window
  - Filter menu size to [2, 50]
  - Group per user; keep users with >= MIN_OBS_MENU sessions

Targets: High Engagement (top tercile of session count in target window)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl

from prefgraph.core.session import MenuChoiceLog

from case_studies.benchmarks.config import (
    TRAIN_FRACTION,
    MIN_OBS_MENU,
    MIN_TRAIN_MENU,
    MIN_TEST_MENU,
)
from case_studies.benchmarks.core.features import (
    extract_menu_baseline,
    extract_menu_rp,
)
from case_studies.benchmarks.core.evaluation import run_three_way, BenchmarkResult


DATASET_NAME = "Taobao (Buy Window)"


def _split_menu_log(log: MenuChoiceLog, fraction: float) -> Tuple[MenuChoiceLog, MenuChoiceLog]:
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


def _find_data_dir(data_dir: str | Path | None) -> Path:
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    env = Path.home() / ".prefgraph" / "data" / "taobao"
    candidates.extend([
        env,
        Path(__file__).resolve().parents[3] / "datasets" / "taobao" / "data",
    ])
    for d in candidates:
        if d.is_dir() and (d / "UserBehavior.csv").exists():
            return d
    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Taobao data not found. Searched:\n  {searched}")


def _load_sample_csv(csv_path: Path, n_rows: int) -> pl.DataFrame:
    df = pl.read_csv(
        csv_path,
        has_header=False,
        new_columns=["user_id", "item_id", "category_id", "behavior_type", "timestamp"],
        n_rows=n_rows,
        schema_overrides={
            "user_id": pl.Int64,
            "item_id": pl.Int64,
            "category_id": pl.Int64,
            "behavior_type": pl.Utf8,
            "timestamp": pl.Int64,
        },
    )
    df = df.filter(pl.col("behavior_type").is_in(["pv", "buy"]))
    return df.select(["user_id", "item_id", "behavior_type", "timestamp"]).sort(["user_id", "timestamp"])  # noqa: E501


def _build_buy_window_logs(
    df: pl.DataFrame,
    window_seconds: int,
    min_menu_size: int = 2,
    max_menu_size: int = 50,
    min_sessions: int = MIN_OBS_MENU,
    max_users: int | None = None,
) -> tuple[Dict[str, MenuChoiceLog], dict[int, list[dict]]]:
    user = df["user_id"].to_numpy()
    item = df["item_id"].to_numpy()
    btype = df["behavior_type"].to_numpy()
    ts = df["timestamp"].to_numpy()

    # Enumerate per-user segments
    segs: List[tuple[int, int, int]] = []
    s = 0
    for i in range(1, len(user)):
        if user[i] != user[i - 1]:
            segs.append((int(user[i - 1]), s, i))
            s = i
    segs.append((int(user[-1]), s, len(user)))

    events_per_user: dict[int, list[tuple[frozenset[int], int]]] = {}
    stats_per_user: dict[int, list[dict]] = {}

    for uid, s, e in segs:
        u_ts = ts[s:e]
        u_item = item[s:e]
        u_type = btype[s:e]

        left = 0
        from collections import Counter, deque
        view_counts: Counter[int] = Counter()
        pv_deque: deque[tuple[int, int]] = deque()  # (ts, item) for PVs only
        ev: list[tuple[frozenset[int], int]] = []
        ev_stats: list[dict] = []
        CAP_GAP = 300  # seconds per-gap cap for active time
        for j in range(e - s):
            t = u_ts[j]
            # Slide left edge to keep [t - W, t)
            while left < j and u_ts[left] < t - window_seconds:
                if u_type[left] == "pv":
                    it = int(u_item[left])
                    view_counts[it] -= 1
                    if view_counts[it] <= 0:
                        del view_counts[it]
                    # Also pop from deque if matches leftmost
                    if pv_deque and pv_deque[0][0] == u_ts[left] and pv_deque[0][1] == it:
                        pv_deque.popleft()
                left += 1

            if u_type[j] == "buy":
                choice = int(u_item[j])
                menu_items = set(view_counts.keys())
                if choice not in menu_items:
                    continue
                if not (min_menu_size <= len(menu_items) <= max_menu_size):
                    continue
                ev.append((frozenset(menu_items), choice))
                # Stats for targets
                pre_pv_total = len(pv_deque)
                # Active time: sum of capped gaps across PVs and final gap to buy
                if pre_pv_total > 0:
                    # PV timestamps are already in time order
                    times = [ts for (ts, _) in pv_deque]
                    # Include buy time as last point
                    times.append(t)
                    gaps = [max(0, min(CAP_GAP, times[k] - times[k - 1])) for k in range(1, len(times))]
                    active_time = int(sum(gaps))
                    # Conversion latency: time from last PV to buy
                    conv_latency = max(0, t - times[-2])
                else:
                    active_time = 0
                    conv_latency = 0
                ev_stats.append({
                    "pre_pv_total": pre_pv_total,
                    "active_time": active_time,
                    "conv_latency": conv_latency,
                })
            else:
                it = int(u_item[j])
                view_counts[it] += 1
                pv_deque.append((u_ts[j], it))

        if len(ev) >= min_sessions:
            events_per_user[uid] = ev
            stats_per_user[uid] = ev_stats

    # Sort by activity and cap users
    user_ids = sorted(events_per_user.keys(), key=lambda u: len(events_per_user[u]), reverse=True)
    if max_users is not None:
        user_ids = user_ids[:max_users]

    # Pack into MenuChoiceLog per user
    logs: Dict[str, MenuChoiceLog] = {}
    for uid in user_ids:
        ev = events_per_user[uid]
        # Remap items per-user to compact integers
        all_items = set()
        for m, _ in ev:
            all_items |= set(m)
        item_map = {it: k for k, it in enumerate(sorted(all_items))}
        menus = [frozenset(item_map[i] for i in m) for m, _ in ev]
        choices = [item_map[c] for _, c in ev]
        logs[str(uid)] = MenuChoiceLog(menus=menus, choices=choices)

    return logs, stats_per_user


def load_and_prepare(
    data_dir: str | Path | None = None,
    *,
    window_seconds: int = 21_600,
    n_rows: int = 2_000_000,
    max_users: int | None = 2000,
):
    import time as _time
    import tracemalloc

    _t_load = _time.perf_counter()
    data_path = _find_data_dir(data_dir)
    df = _load_sample_csv(data_path / "UserBehavior.csv", n_rows)
    logs, stats_per_user = _build_buy_window_logs(
        df,
        window_seconds=window_seconds,
        min_sessions=MIN_OBS_MENU,
        max_users=max_users,
    )
    load_and_prepare.load_time_s = _time.perf_counter() - _t_load

    train_logs, user_ids = {}, []
    targets = {
        "high_engagement": [],
        "concentration": [],
        "novelty": [],
        "entropy": [],
        "drift": [],
        "click_volume": [],
        "active_time": [],
        "median_latency": [],
    }

    for uid, log in logs.items():
        T = len(log.choices)
        if T < MIN_OBS_MENU:
            continue
        split = int(T * TRAIN_FRACTION)
        if split < MIN_TRAIN_MENU or (T - split) < MIN_TEST_MENU:
            continue

        train_log, test_log = _split_menu_log(log, TRAIN_FRACTION)
        train_logs[uid] = train_log
        user_ids.append(uid)
        # Engagement: count of sessions in test window
        targets["high_engagement"].append(len(test_log.choices))
        # Choice concentration in test window (modal share). Low = low loyalty
        from collections import Counter as _Ctr
        tc = _Ctr(test_log.choices)
        targets["concentration"].append(tc.most_common(1)[0][1] / max(len(test_log.choices), 1))
        # Novelty: fraction of unique test choices not seen in train
        train_items = set(train_log.choices)
        test_items = set(test_log.choices)
        novelty = (len(test_items - train_items) / max(len(test_items), 1)) if test_items else 0.0
        targets["novelty"].append(novelty)
        # Entropy: normalized choice entropy in test window
        import math as _m
        probs = np.array(list(tc.values())) / max(len(test_log.choices), 1)
        if len(probs) > 0:
            ent = float(-np.sum(probs * np.log2(probs + 1e-12)))
            norm_ent = ent / max(np.log2(len(probs)), 1e-12)
        else:
            norm_ent = 0.0
        targets["entropy"].append(norm_ent)
        # Preference drift: dispersion increase (unique ratio test > train)
        train_unique_ratio = len(set(train_log.choices)) / max(len(train_log.choices), 1)
        test_unique_ratio = len(set(test_log.choices)) / max(len(test_log.choices), 1)
        targets["drift"].append(1 if test_unique_ratio > train_unique_ratio else 0)

        # Click volume, active time, latency from event stats
        # Align by splitting the stats list by the same split index
        ev_stats = stats_per_user[int(uid)]
        split_idx = int(len(ev_stats) * TRAIN_FRACTION)
        test_stats = ev_stats[split_idx:]
        if test_stats:
            click_sum = int(sum(s["pre_pv_total"] for s in test_stats))
            active_sum = int(sum(s["active_time"] for s in test_stats))
            latencies = [s["conv_latency"] for s in test_stats if s["conv_latency"] > 0]
            median_latency = float(np.median(latencies)) if latencies else 0.0
        else:
            click_sum = 0
            active_sum = 0
            median_latency = 0.0
        targets["click_volume"].append(click_sum)
        targets["active_time"].append(active_sum)
        targets["median_latency"].append(median_latency)

    if not user_ids:
        return None, None, {}, []

    engagement = np.array(targets["high_engagement"])
    concentration = np.array(targets["concentration"])  # higher = more loyal
    novelty = np.array(targets["novelty"])  # higher = more novel
    entropy = np.array(targets["entropy"])  # higher = more dispersed choices
    drift = np.array(targets["drift"]).astype(int)
    click_volume = np.array(targets["click_volume"]).astype(float)
    active_time = np.array(targets["active_time"]).astype(float)
    median_latency = np.array(targets["median_latency"]).astype(float)

    from case_studies.benchmarks.core.eda import compute_menu_eda
    load_and_prepare.eda = compute_menu_eda(train_logs)

    X_base = extract_menu_baseline(train_logs)

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

    targets_dict = {
        "High Engagement": (
            (engagement > np.percentile(engagement, 66.67)).astype(int),
            "classification", engagement, 66.67,
        ),
        "Low Loyalty": (
            # Low loyalty = low concentration (bottom third of concentration)
            (concentration < np.percentile(concentration, 33.33)).astype(int),
            "classification", 1.0 - concentration, 66.67,
        ),
        "High Novelty": (
            (novelty > np.percentile(novelty, 66.67)).astype(int),
            "classification", novelty, 66.67,
        ),
        "Any Novelty": (
            (novelty > 0).astype(int),
            "classification", novelty, 66.67,
        ),
        "High Entropy": (
            (entropy > np.percentile(entropy, 66.67)).astype(int),
            "classification", entropy, 66.67,
        ),
        "Pref Drift": (
            drift,
            "classification", drift.astype(float), 50.0,
        ),
        "High Click Volume": (
            (click_volume > np.percentile(click_volume, 66.67)).astype(int),
            "classification", click_volume, 66.67,
        ),
        "High Active Time": (
            (active_time > np.percentile(active_time, 66.67)).astype(int),
            "classification", active_time, 66.67,
        ),
        "Fast Conversion": (
            # Bottom tercile of median per-user latency ⇒ fast
            (median_latency < np.percentile(median_latency, 33.33)).astype(int),
            "classification", -median_latency, 66.67,
        ),
    }
    return X_rp, X_base, targets_dict, user_ids


def run_benchmark(
    data_dir: str | Path | None = None,
    *,
    window_seconds: int = 21_600,
    n_rows: int = 2_000_000,
    max_users: int | None = 2000,
) -> list[BenchmarkResult]:
    X_rp, X_base, targets_dict, user_ids = load_and_prepare(data_dir, window_seconds=window_seconds, n_rows=n_rows, max_users=max_users)

    if X_rp is None:
        return []

    _load_t = getattr(load_and_prepare, "load_time_s", 0.0)
    _engine_t = getattr(load_and_prepare, "engine_time_s", 0.0)
    _feat_t = getattr(load_and_prepare, "feature_time_s", 0.0)
    _mem = getattr(load_and_prepare, "peak_memory_mb", 0.0)

    results = []
    for target_name, (y, task_type, y_cont, pctl) in targets_dict.items():
        try:
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
            print(
                f"  [{DATASET_NAME}] Target: {target_name}  "
                f"AUC Base={result.auc_base:.3f}  +RP={result.auc_combined:.3f}  "
                f"Lift={result.auc_combined - result.auc_base:+.3f}"
            )
        except Exception as e:
            print(f"  [{DATASET_NAME}] Target: {target_name}  SKIP ({e})")

    return results
