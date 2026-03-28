"""Yoochoose RecSys 2015 click-stream dataset loader.

Loads click session data from the RecSys 2015 challenge and reconstructs
menu-choice observations: items clicked in a session = menu, purchased
item = choice.

Data download:
  The dataset is available from the RecSys 2015 challenge archives.
  Place yoochoose-clicks.dat and yoochoose-buys.dat in the data directory.

Source: https://recsys.acm.org/recsys15/challenge/
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from prefgraph.core.session import MenuChoiceLog


SESSION_GAP_MINUTES = 30
MIN_MENU_SIZE = 2
MAX_MENU_SIZE = 50


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find Yoochoose data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "yoochoose")

    candidates.extend([
        Path.home() / ".prefgraph" / "data" / "yoochoose",
        Path(__file__).resolve().parents[3] / "datasets" / "yoochoose" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (
            (d / "yoochoose-clicks.dat").exists() or
            (d / "yoochoose-clicks.csv").exists()
        ):
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Yoochoose data not found. Searched:\n  {searched}\n\n"
        "Download from the RecSys 2015 challenge:\n"
        "  https://recsys.acm.org/recsys15/challenge/\n\n"
        "Required files: yoochoose-clicks.dat, yoochoose-buys.dat\n"
        "Place in one of the directories above."
    )


def load_yoochoose(
    data_dir: str | Path | None = None,
    min_sessions: int = 5,
    max_users: int | None = 5000,
    remap_items: bool = True,
) -> dict[str, MenuChoiceLog]:
    """Load Yoochoose click-stream data as menu-choice observations.

    Each session with a purchase becomes a menu-choice observation:
    items clicked = menu, purchased item = choice.

    Args:
        data_dir: Path to directory containing yoochoose-clicks.dat
            and yoochoose-buys.dat.
        min_sessions: Minimum purchase sessions per user.
        max_users: Cap number of users returned (default 5000).
        remap_items: Remap item IDs to 0..N-1 per user.

    Returns:
        Dict mapping session_id (str) -> MenuChoiceLog.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for dataset loaders. "
            "Install with: pip install 'prefgraph[datasets]'"
        ) from None

    data_path = _find_data_dir(data_dir)

    # Load clicks
    clicks_file = data_path / "yoochoose-clicks.dat"
    if not clicks_file.exists():
        clicks_file = data_path / "yoochoose-clicks.csv"

    print(f"  Loading Yoochoose clicks from {clicks_file}...")
    clicks = pd.read_csv(
        clicks_file,
        names=["session_id", "timestamp", "item_id", "category"],
        parse_dates=["timestamp"],
    )
    print(f"  Raw clicks: {len(clicks):,}")

    # Load buys
    buys_file = data_path / "yoochoose-buys.dat"
    if not buys_file.exists():
        buys_file = data_path / "yoochoose-buys.csv"

    buys = pd.read_csv(
        buys_file,
        names=["session_id", "timestamp", "item_id", "price", "quantity"],
        parse_dates=["timestamp"],
    )
    print(f"  Raw buys: {len(buys):,}")

    # Find sessions with exactly 1 unique purchased item
    buy_sessions = buys.groupby("session_id")["item_id"].nunique()
    single_buy_sessions = set(buy_sessions[buy_sessions == 1].index)

    # Get the purchased item per session
    session_purchases = (
        buys[buys["session_id"].isin(single_buy_sessions)]
        .groupby("session_id")["item_id"]
        .first()
        .to_dict()
    )

    # Filter clicks to sessions with purchases
    clicks_with_buys = clicks[clicks["session_id"].isin(single_buy_sessions)]

    # Build menus: items clicked in each session
    session_menus = (
        clicks_with_buys
        .groupby("session_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    # Build (menu, choice) pairs
    records = []
    for session_id, menu in session_menus.items():
        choice = session_purchases.get(session_id)
        if choice is None:
            continue
        menu = menu | {choice}  # Ensure choice is in menu
        if len(menu) < MIN_MENU_SIZE or len(menu) > MAX_MENU_SIZE:
            continue
        records.append({
            "session_id": session_id,
            "menu": frozenset(menu),
            "choice": choice,
        })

    print(f"  Valid sessions: {len(records):,}")

    # Yoochoose sessions are anonymous - group by session_id patterns
    # Each session_id is a unique user visit. To get per-"user" data,
    # we need multiple sessions from the same user. Yoochoose doesn't
    # have persistent user IDs, so we use session_id as user_id directly
    # (each session = one user with 1 observation is not useful for RP analysis).
    #
    # Alternative: group sessions that share many items as likely same user.
    # For simplicity, we create synthetic "users" by grouping consecutive
    # sessions that share overlapping items.
    #
    # SIMPLIFIED APPROACH: Since each session is one observation, we sample
    # random groups of sessions to form "synthetic users" for RP analysis.
    # This tests whether a random sample of shoppers shows consistent
    # preferences across the catalog.

    # Group sessions by the most-purchased item category to create user proxies
    df = pd.DataFrame(records)

    # Use category from clicks to group sessions
    click_categories = (
        clicks_with_buys[clicks_with_buys["session_id"].isin(df["session_id"])]
        .groupby("session_id")["category"]
        .first()
        .to_dict()
    )
    df["category"] = df["session_id"].map(click_categories)

    # Group by category and create users from batches of sessions
    user_logs: dict[str, MenuChoiceLog] = {}
    user_count = 0

    for cat, group in df.groupby("category"):
        if len(group) < min_sessions:
            continue

        # Split this category's sessions into user-sized chunks
        sessions_list = group.to_dict("records")
        for chunk_start in range(0, len(sessions_list), min_sessions * 2):
            chunk = sessions_list[chunk_start:chunk_start + min_sessions * 2]
            if len(chunk) < min_sessions:
                continue

            menus = [r["menu"] for r in chunk]
            choices = [r["choice"] for r in chunk]

            if remap_items:
                all_items = set()
                for m in menus:
                    all_items |= m
                item_map = {item: idx for idx, item in enumerate(sorted(all_items))}
                menus = [frozenset(item_map[i] for i in m) for m in menus]
                choices = [item_map[c] for c in choices]

            uid = f"user_{user_count}"
            user_logs[uid] = MenuChoiceLog(menus=menus, choices=choices)
            user_count += 1

            if max_users is not None and user_count >= max_users:
                break
        if max_users is not None and user_count >= max_users:
            break

    print(f"  Built {len(user_logs)} MenuChoiceLog objects")
    return user_logs
