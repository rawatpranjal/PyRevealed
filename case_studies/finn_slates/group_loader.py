"""Group-level data loader for FINN.no Slates.

Maps individual item IDs to their 290 category-geography groups using
itemattr.npz, producing dense preference graphs suitable for SARP/WARP
and stochastic choice analysis.

Individual FINN.no listings are unique (1.3M items, ~6% repeat rate),
making item-level preference graphs too sparse for meaningful RP analysis.
At the group level (290 groups like "JOB, Rogaland" or "BAP, antiques,
Trondlag"), the same categories appear across many slates, giving
testable preference relations.

Uses DuckDB for the heavy data transformation (flattening 2.3M users x
20 timesteps x 25 slate slots into group-level menus) and numpy for the
initial item-to-group mapping.

Reference:
  Eide et al. (2021). RecSys 2021. arxiv.org/abs/2111.03340
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np

from prefgraph.core.session import MenuChoiceLog, StochasticChoiceLog
from prefgraph.datasets._finn_slates import (
    _find_data_dir,
    PADDING_THRESHOLD,
)

# Groups 0 (PAD), 1 (noClick), 2 (UNK) are special tokens
GROUP_THRESHOLD = 3


def _load_item_groups(data_dir: Path) -> tuple[np.ndarray, dict[int, str]]:
    """Load item-to-group mapping and group labels.

    Returns:
        item_groups: int array [n_items] mapping item_id -> group_id (0..289).
        group_labels: dict mapping group_id -> label (e.g. "JOB, Rogaland").
    """
    attr_path = data_dir / "itemattr.npz"
    if not attr_path.exists():
        raise FileNotFoundError(
            f"itemattr.npz not found at {attr_path}. "
            "Download from: https://drive.google.com/uc?id=1rKKyMQZqWp8vQ-Pl1SeHrQxzc5dXldnR"
        )

    with np.load(attr_path, allow_pickle=True) as f:
        item_groups = f[list(f.keys())[0]]

    if item_groups.ndim == 2:
        item_groups = item_groups[:, 0]
    item_groups = item_groups.astype(np.int32)
    n_groups = int(item_groups.max()) + 1
    print(f"  Item groups: {len(item_groups):,} items -> {n_groups} groups")

    # ind2val.json: {"category": {"0": "PAD", ...}, "interaction_type": {...}}
    group_labels: dict[int, str] = {}
    ind2val_path = data_dir / "ind2val.json"
    if ind2val_path.exists():
        with open(ind2val_path) as f:
            ind2val = json.load(f)
        cat_dict = ind2val.get("category", ind2val)
        for k, v in cat_dict.items():
            try:
                idx = int(k)
                if 0 <= idx < n_groups:
                    group_labels[idx] = v
            except ValueError:
                pass
        print(f"  Group labels loaded: {len(group_labels)} entries")
        for gid, label in [(g, l) for g, l in sorted(group_labels.items()) if g >= 3][:3]:
            print(f"    Group {gid}: {label}")

    return item_groups, group_labels


def load_group_level(
    data_dir: str | Path | None = None,
    min_sessions: int = 5,
    max_users: int | None = 100_000,
) -> tuple[dict[str, MenuChoiceLog], dict[int, str], dict]:
    """Load FINN.no data with items mapped to 290 category-geography groups.

    Uses numpy for vectorized item-to-group mapping and DuckDB for the
    flattening, filtering, deduplication, and per-user aggregation.

    Args:
        data_dir: Path to FINN.no data directory (auto-detected if None).
        min_sessions: Minimum click events per user.
        max_users: Cap on number of users to load.

    Returns:
        user_logs: Dict mapping uid -> group-level MenuChoiceLog.
        group_labels: Dict mapping group_id -> human-readable label.
        stats: Dict with loading statistics.
    """
    data_path = _find_data_dir(data_dir)
    item_groups, group_labels = _load_item_groups(data_path)

    npz_path = data_path / "data.npz"
    print(f"  Loading raw arrays from {npz_path}...")

    # Load click + interaction_type first (small arrays), then slate
    # separately. Loading all three in one `with` block keeps the full
    # decompressed npz in memory; separate loads let GC reclaim between.
    with np.load(npz_path, allow_pickle=False) as d:
        click = d["click"]               # [N_full, T]
        interaction_type = d["interaction_type"]
    N_full, T = click.shape
    cap = min(max_users, N_full) if max_users else N_full
    click = click[:cap]
    interaction_type = interaction_type[:cap]
    print(f"  Raw: {N_full:,} users x {T} timesteps, loading slate for {cap:,}...")

    with np.load(npz_path, allow_pickle=False) as d:
        slate = d["slate"][:cap]         # [cap, T, K]
    K = slate.shape[2]
    N = cap
    print(f"  Loaded: {N:,} users x {T} timesteps x {K} slate slots")

    # ---- Step 1: Vectorized item -> group mapping (numpy) ----
    max_item_id = len(item_groups) - 1
    click_clamped = np.clip(click, 0, max_item_id)
    click_group = item_groups[click_clamped].astype(np.int16)    # [N, T]

    slate_group = item_groups[
        np.clip(slate.reshape(-1), 0, max_item_id)
    ].reshape(N, T, K).astype(np.int16)  # [N, T, K]

    # ---- Step 2: Flatten valid observations (numpy) ----
    print(f"  Building observation table...")
    valid_click = (click >= PADDING_THRESHOLD) & (click_group >= GROUP_THRESHOLD)
    user_idx, time_idx = np.where(valid_click)

    choice_groups = click_group[user_idx, time_idx]
    itypes = interaction_type[user_idx, time_idx]
    obs_slates = slate_group[user_idx, time_idx]  # [n_valid, K]

    n_valid = len(user_idx)
    print(f"  Valid observations: {n_valid:,}")

    # ---- Step 3: DuckDB for menu building + aggregation ----
    # Register flat arrays as a DuckDB table. Each slate slot is a column.
    con = duckdb.connect()

    # Build a dict of columns for the base observation table
    cols = {
        "uid": user_idx.astype(np.int32),
        "t": time_idx.astype(np.int16),
        "choice_group": choice_groups,
        "itype": itypes.astype(np.int8),
    }
    for k in range(K):
        cols[f"s{k}"] = obs_slates[:, k]

    con.execute("CREATE TABLE obs AS SELECT * FROM cols")

    # Build menus: UNNEST the 25 slate columns + the choice group into a
    # single column, filter to real groups, deduplicate, sort, re-aggregate.
    # This runs entirely inside DuckDB's vectorized engine.
    slate_union = " UNION ALL ".join(
        f"SELECT uid, t, s{k} AS g FROM obs WHERE s{k} >= {GROUP_THRESHOLD}"
        for k in range(K)
    )
    # Also include choice_group in the menu
    slate_union += f" UNION ALL SELECT uid, t, choice_group AS g FROM obs"

    print(f"  Building group-level menus in DuckDB...")
    con.execute(f"""
        CREATE TABLE menus AS
        SELECT uid, t, list_sort(list_distinct(list(g))) AS menu_groups
        FROM ({slate_union})
        GROUP BY uid, t
        HAVING len(list_distinct(list(g))) >= 2
    """)

    n_menus = con.execute("SELECT count(*) FROM menus").fetchone()[0]
    print(f"  After menu-size filter: {n_menus:,} observations")

    # Join back to get choice_group and itype per observation
    con.execute("""
        CREATE TABLE full_obs AS
        SELECT m.uid, m.t, o.choice_group, o.itype, m.menu_groups
        FROM menus m
        JOIN obs o ON m.uid = o.uid AND m.t = o.t
        ORDER BY m.uid, m.t
    """)

    # Filter to users with >= min_sessions observations
    con.execute(f"""
        CREATE TABLE qualifying AS
        SELECT uid FROM full_obs GROUP BY uid HAVING count(*) >= {min_sessions}
    """)
    n_qualifying = con.execute("SELECT count(*) FROM qualifying").fetchone()[0]
    print(f"  Qualifying users (>= {min_sessions} obs): {n_qualifying:,}")

    # Fetch final data grouped by user
    rows = con.execute("""
        SELECT f.uid,
               list(f.choice_group ORDER BY f.t) AS choices,
               list(f.menu_groups ORDER BY f.t) AS menus,
               list(f.itype ORDER BY f.t) AS itypes
        FROM full_obs f
        JOIN qualifying q ON f.uid = q.uid
        GROUP BY f.uid
        ORDER BY f.uid
    """).fetchall()

    con.close()

    # ---- Step 4: Build MenuChoiceLogs ----
    print(f"  Building {len(rows):,} MenuChoiceLogs...")
    user_logs: dict[str, MenuChoiceLog] = {}
    user_itypes: dict[str, list[int]] = {}
    n_kept_obs = 0

    for uid, choices_list, menus_list, itypes_list in rows:
        menus_orig = [frozenset(int(g) for g in m) for m in menus_list]
        choices_orig = [int(c) for c in choices_list]

        # Remap to compact 0..N-1 per user
        all_groups = sorted(set().union(*menus_orig))
        group_map = {g: idx for idx, g in enumerate(all_groups)}
        reverse_map = {idx: g for g, idx in group_map.items()}

        user_logs[str(uid)] = MenuChoiceLog(
            menus=[frozenset(group_map[g] for g in m) for m in menus_orig],
            choices=[group_map[c] for c in choices_orig],
            metadata={"group_reverse_map": reverse_map},
        )
        user_itypes[str(uid)] = [int(t) for t in itypes_list]
        n_kept_obs += len(choices_list)

    stats = {
        "n_users_raw": N,
        "n_users": len(user_logs),
        "n_valid_obs": n_valid,
        "n_kept_obs": n_kept_obs,
        "interaction_types": user_itypes,
    }

    print(f"  Built {len(user_logs):,} MenuChoiceLogs ({n_kept_obs:,} observations)")
    return user_logs, group_labels, stats


def build_stochastic_logs(
    user_logs: dict[str, MenuChoiceLog],
    min_repeated_menus: int = 3,
    min_total_obs_per_menu: int = 2,
) -> dict[str, StochasticChoiceLog]:
    """Convert group-level MenuChoiceLogs to StochasticChoiceLogs.

    For each user, finds menus that appear multiple times and builds
    choice frequency distributions from repeated presentations.
    """
    stochastic_logs: dict[str, StochasticChoiceLog] = {}
    n_skipped = 0

    for uid, log in user_logs.items():
        menu_counts: dict[frozenset[int], dict[int, int]] = {}
        menu_totals: dict[frozenset[int], int] = {}

        for menu, choice in zip(log.menus, log.choices):
            if menu not in menu_counts:
                menu_counts[menu] = Counter()
                menu_totals[menu] = 0
            menu_counts[menu][choice] += 1
            menu_totals[menu] += 1

        repeated_menus = [
            m for m, total in menu_totals.items()
            if total >= min_total_obs_per_menu
        ]

        if len(repeated_menus) < min_repeated_menus:
            n_skipped += 1
            continue

        stochastic_logs[uid] = StochasticChoiceLog(
            menus=list(repeated_menus),
            choice_frequencies=[dict(menu_counts[m]) for m in repeated_menus],
            total_observations_per_menu=[menu_totals[m] for m in repeated_menus],
            user_id=uid,
        )

    print(f"  Stochastic logs: {len(stochastic_logs):,} users "
          f"(skipped {n_skipped:,} with < {min_repeated_menus} repeated menus)")
    return stochastic_logs


def compute_search_ratios(stats: dict) -> dict[str, float]:
    """Compute search-vs-recommendation ratio for each user.

    Returns dict mapping uid -> fraction of interactions from search (type 1).
    """
    itypes = stats.get("interaction_types", {})
    ratios: dict[str, float] = {}
    for uid, types in itypes.items():
        if types:
            ratios[uid] = sum(1 for t in types if t == 1) / len(types)
        else:
            ratios[uid] = 0.0
    return ratios
