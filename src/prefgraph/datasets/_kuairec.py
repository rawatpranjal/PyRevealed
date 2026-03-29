"""KuaiRec near-dense interaction dataset loader.

Loads the KuaiRec big_matrix.csv (1,411 users × 3,327 videos, ~99% density)
and reconstructs daily menu-choice observations for revealed preference analysis.

Menu construction (daily aggregation):
  - Group all interactions by (user_id, date) — each calendar day is one session.
  - menu   = frozenset of all video_ids the user watched that day.
  - choice = video_id with the highest watch_ratio that day (rewatched video is
             most preferred, consistent with revealed preference theory).
  - Skip days where the resulting menu has fewer than min_menu_size items — a
    singleton menu cannot inform preference (no trade-off is revealed).

watch_ratio semantics (from KuaiRec paper):
  - watch_ratio = duration_watched / video_duration
  - watch_ratio > 1.0 means the user re-watched (strong engagement signal)
  - watch_ratio in (0, 1) means partial watch
  - watch_ratio == 0 means the video was shown but not meaningfully watched

Dataset: KuaiRec (Gao et al., 2022)
  "KuaiRec: A Fully-Observed Dataset and Insights for Evaluating Recommender Systems"
  CIKM 2022. https://arxiv.org/abs/2202.10842

Download:
  https://github.com/chongminggao/KuaiRec
  Get big_matrix.csv (requires Git LFS: git lfs pull)
  Place big_matrix.csv in one of the cascade directories below.

License: MIT (code). Data: KuaiRec data license (research only).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl

from prefgraph.core.session import MenuChoiceLog


# --- Constants ---

MIN_MENU_SIZE = 2    # Minimum items watched in a day to form a valid menu
MAX_MENU_SIZE = 50   # Cap menu size; dense dataset can produce very large daily menus
MIN_SESSIONS_PER_USER = 5  # User must have at least this many qualifying days


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find KuaiRec data directory using the standard cascade.

    Search order:
      1. explicit data_dir argument
      2. $PYREVEALED_DATA_DIR/kuairec
      3. ~/.prefgraph/data/kuairec
      4. <repo_root>/datasets/kuairec (dev convenience)
    """
    candidates: list[Path] = []

    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "kuairec")

    candidates.extend([
        Path.home() / ".prefgraph" / "data" / "kuairec",
        Path(__file__).resolve().parents[3] / "datasets" / "kuairec",
    ])

    for d in candidates:
        if d.is_dir() and (d / "big_matrix.csv").exists():
            # Verify the file is non-empty (Git LFS stubs are 0 bytes)
            csv_path = d / "big_matrix.csv"
            if csv_path.stat().st_size > 0:
                return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"KuaiRec data not found. Searched:\n  {searched}\n\n"
        "Download from: https://github.com/chongminggao/KuaiRec\n"
        "Get big_matrix.csv (use Git LFS download) and place in one of the above directories.\n"
        "  git lfs install\n"
        "  git lfs pull  # inside the KuaiRec repo clone\n"
        "  cp big_matrix.csv ~/.prefgraph/data/kuairec/"
    )


def load_kuairec(
    data_dir: str | Path | None = None,
    min_sessions: int = MIN_SESSIONS_PER_USER,
    max_users: int | None = None,
    remap_items: bool = True,
) -> dict[str, MenuChoiceLog]:
    """Load KuaiRec interaction data as daily menu-choice observations.

    Groups each user's interactions by calendar date. Within each day:
      - All watched videos form the menu (revealed consideration set).
      - The video with the highest watch_ratio is the revealed choice.

    This construction exploits KuaiRec's near-100% density: on any given day
    a user is shown a large representative set of videos, so the daily set of
    watches approximates an explicit menu.

    Args:
        data_dir: Path to directory containing big_matrix.csv.
            If None, searches standard locations (see _find_data_dir).
        min_sessions: Minimum qualifying days per user (default: 5).
            Users with fewer days are dropped.
        max_users: Cap on number of users returned. If None, all qualifying
            users are returned (only 1,411 users total in KuaiRec).
        remap_items: If True, remap item IDs to 0..N-1 per user for
            compact integer representation expected by MenuChoiceLog.

    Returns:
        Dict mapping user_id (str) -> MenuChoiceLog.

    Raises:
        FileNotFoundError: If big_matrix.csv is not found or is a Git LFS stub.
    """
    data_path = _find_data_dir(data_dir)
    csv_file = data_path / "big_matrix.csv"

    print(f"  Loading KuaiRec interactions from {csv_file}...")

    # -------------------------------------------------------------------------
    # Step 1: Read the interaction matrix with polars.
    #
    # Expected columns: user_id, video_id, watch_ratio, date
    # The date column is a calendar date string (e.g. "2020-08-22").
    # We keep only the 4 columns we need; drop all others (duration, etc.)
    # -------------------------------------------------------------------------
    df = pl.read_csv(
        csv_file,
        infer_schema_length=10000,
    )

    print(f"  Raw interactions: {len(df):,}")

    # Detect actual column names (the dataset may include extra columns)
    cols = df.columns
    needed = [c for c in ["user_id", "video_id", "watch_ratio", "date"] if c in cols]
    if len(needed) < 4:
        # Try alternative spellings
        rename_map = {}
        for c in cols:
            cl = c.lower().strip()
            if cl in ("user_id", "userid"):
                rename_map[c] = "user_id"
            elif cl in ("video_id", "videoid", "item_id"):
                rename_map[c] = "video_id"
            elif cl in ("watch_ratio", "watch ratio", "ratio"):
                rename_map[c] = "watch_ratio"
            elif cl in ("date", "day"):
                rename_map[c] = "date"
        if rename_map:
            df = df.rename(rename_map)

    # Ensure all required columns exist after possible rename
    required = {"user_id", "video_id", "watch_ratio", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"KuaiRec big_matrix.csv is missing columns: {missing}.\n"
            f"Found columns: {df.columns}"
        )

    # Cast types for safety
    df = df.select([
        pl.col("user_id").cast(pl.Int64),
        pl.col("video_id").cast(pl.Int64),
        pl.col("watch_ratio").cast(pl.Float64),
        pl.col("date").cast(pl.Utf8),  # Keep as string date for grouping
    ])

    # -------------------------------------------------------------------------
    # Step 2: Group by (user_id, date) to form daily menus.
    #
    # For each (user, day) group:
    #   - menu = all video_ids watched (frozenset for MenuChoiceLog)
    #   - choice = video_id with the maximum watch_ratio
    #     (ties broken by lower video_id for determinism)
    #
    # Menu size filter: only keep days where menu_size >= min_menu_size.
    # This removes "solo browse" days where only 1 video was watched —
    # a singleton menu gives zero preference information.
    # -------------------------------------------------------------------------
    print("  Building daily menus (group by user_id × date)...")

    # Sort so argmax is deterministic on ties (lower video_id wins)
    df = df.sort(["user_id", "date", "video_id"])

    # Group: collect (video_ids, watch_ratios) per (user, date)
    grouped = df.group_by(["user_id", "date"]).agg([
        pl.col("video_id").alias("video_ids"),
        pl.col("watch_ratio").alias("watch_ratios"),
    ])

    # Convert to Python lists for MenuChoiceLog construction
    # (polars list columns → Python objects)
    records: list[dict] = []
    for row in grouped.iter_rows(named=True):
        vids = row["video_ids"]        # list[int]
        ratios = row["watch_ratios"]   # list[float]

        menu_size = len(vids)
        if menu_size < MIN_MENU_SIZE or menu_size > MAX_MENU_SIZE:
            continue

        # Revealed choice = argmax watch_ratio (already sorted by video_id so ties pick lower id)
        best_idx = int(np.argmax(ratios))
        choice_vid = vids[best_idx]

        records.append({
            "user_id": row["user_id"],
            "date": row["date"],
            "menu": frozenset(vids),
            "choice": choice_vid,
        })

    total_sessions = len(records)
    print(f"  Qualifying daily sessions (menu size {MIN_MENU_SIZE}-{MAX_MENU_SIZE}): {total_sessions:,}")

    # -------------------------------------------------------------------------
    # Step 3: Group records by user_id, apply min_sessions filter.
    # -------------------------------------------------------------------------
    from collections import defaultdict
    user_records: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        user_records[rec["user_id"]].append(rec)

    # Keep only users with at least min_sessions qualifying days
    qualifying: dict[int, list[dict]] = {
        uid: recs for uid, recs in user_records.items()
        if len(recs) >= min_sessions
    }

    # Sort descending by session count for deterministic top-k slicing
    qualifying = dict(
        sorted(qualifying.items(), key=lambda kv: len(kv[1]), reverse=True)
    )

    print(f"  Users with >= {min_sessions} qualifying days: {len(qualifying):,}")

    if max_users is not None and len(qualifying) > max_users:
        qualifying = dict(list(qualifying.items())[:max_users])
        print(f"  Capped to {max_users} users")

    # -------------------------------------------------------------------------
    # Step 4: Build MenuChoiceLog per user.
    #
    # MenuChoiceLog expects:
    #   menus:   list[frozenset[int]]
    #   choices: list[int]
    #
    # Sort each user's records by date so observations are temporally ordered —
    # the benchmark train/test split relies on temporal ordering.
    # -------------------------------------------------------------------------
    user_logs: dict[str, MenuChoiceLog] = {}

    for uid, recs in qualifying.items():
        # Sort by date for temporal ordering
        recs_sorted = sorted(recs, key=lambda r: r["date"])
        menus = [r["menu"] for r in recs_sorted]
        choices = [r["choice"] for r in recs_sorted]

        if remap_items:
            # Remap all item IDs encountered by this user to 0..N-1.
            # This produces compact integer arrays that Rust can work with
            # efficiently, and avoids sparse large IDs (3,327 videos → dense).
            all_items: set[int] = set()
            for m in menus:
                all_items |= m
            item_map = {item: idx for idx, item in enumerate(sorted(all_items))}
            menus = [frozenset(item_map[v] for v in m) for m in menus]
            choices = [item_map[c] for c in choices]

        user_logs[str(uid)] = MenuChoiceLog(menus=menus, choices=choices)

    print(f"  Built {len(user_logs)} MenuChoiceLog objects")
    return user_logs


def get_kuairec_summary(user_logs: dict[str, MenuChoiceLog]) -> dict:
    """Get summary statistics for loaded KuaiRec data."""
    if not user_logs:
        return {"n_users": 0}

    sessions_per_user = [len(log.choices) for log in user_logs.values()]
    menu_sizes: list[int] = []
    for log in user_logs.values():
        menu_sizes.extend(len(m) for m in log.menus)

    return {
        "n_users": len(user_logs),
        "total_sessions": sum(sessions_per_user),
        "mean_sessions": float(np.mean(sessions_per_user)),
        "median_sessions": float(np.median(sessions_per_user)),
        "mean_menu_size": float(np.mean(menu_sizes)),
        "median_menu_size": float(np.median(menu_sizes)),
    }
