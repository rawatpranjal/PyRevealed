"""Taobao User Behavior dataset loader.

Loads the Taobao/Alibaba user behavior dataset (100M events, 988K users)
and reconstructs daily menu-choice observations: items viewed in a day
form the menu, purchased item is the choice.

Data must be downloaded from Kaggle:
  kaggle datasets download -d marwa80/userbehavior
  unzip userbehavior.zip -d ~/.pyrevealed/data/taobao/

Source: https://www.kaggle.com/datasets/marwa80/userbehavior
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pyrevealed.core.session import MenuChoiceLog


MIN_MENU_SIZE = 2
MAX_MENU_SIZE = 50
CHUNK_SIZE = 5_000_000


def _find_data_dir(data_dir: str | Path | None) -> Path:
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "taobao")

    candidates.extend([
        Path.home() / ".pyrevealed" / "data" / "taobao",
        Path(__file__).resolve().parents[3] / "datasets" / "taobao" / "data",
    ])

    for d in candidates:
        if d.is_dir() and (d / "UserBehavior.csv").exists():
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Taobao data not found. Searched:\n  {searched}\n\n"
        "Download from Kaggle:\n"
        "  kaggle datasets download -d marwa80/userbehavior\n"
        "  unzip userbehavior.zip -d ~/.pyrevealed/data/taobao/"
    )


def load_taobao(
    data_dir: str | Path | None = None,
    min_sessions: int = 5,
    max_users: int | None = 50000,
    remap_items: bool = True,
) -> dict[str, MenuChoiceLog]:
    """Load Taobao user behavior as menu-choice observations.

    Daily sessions: items viewed (pv) = menu, item bought (buy) = choice.
    Only keeps days with exactly 1 purchase and >= 2 viewed items.

    Args:
        data_dir: Path to directory containing UserBehavior.csv.
        min_sessions: Minimum purchase-days per user.
        max_users: Cap on number of users returned.
        remap_items: Remap item IDs to 0..N-1 per user.

    Returns:
        Dict mapping user_id (str) -> MenuChoiceLog.
    """
    data_path = _find_data_dir(data_dir)
    csv_file = data_path / "UserBehavior.csv"

    print(f"  Loading Taobao events from {csv_file} (chunked)...")

    # Read in chunks, filter to pv + buy only
    all_views = []  # (user_id, date, item_id)
    all_buys = []   # (user_id, date, item_id)

    for chunk in pd.read_csv(
        csv_file,
        header=None,
        names=["user_id", "item_id", "category_id", "behavior_type", "timestamp"],
        usecols=["user_id", "item_id", "behavior_type", "timestamp"],
        dtype={"user_id": "int64", "item_id": "int64", "behavior_type": "category"},
        chunksize=CHUNK_SIZE,
    ):
        # Convert timestamp to date
        chunk["date"] = pd.to_datetime(chunk["timestamp"], unit="s").dt.date

        views = chunk[chunk["behavior_type"] == "pv"][["user_id", "date", "item_id"]]
        buys = chunk[chunk["behavior_type"] == "buy"][["user_id", "date", "item_id"]]

        all_views.append(views)
        all_buys.append(buys)

    views_df = pd.concat(all_views, ignore_index=True)
    buys_df = pd.concat(all_buys, ignore_index=True)
    del all_views, all_buys

    print(f"  Views: {len(views_df):,}, Buys: {len(buys_df):,}")

    # Find (user, date) pairs with exactly 1 unique purchased item
    buy_counts = buys_df.groupby(["user_id", "date"])["item_id"].nunique()
    single_buy_sessions = set(buy_counts[buy_counts == 1].index)

    # Get purchased item per session
    session_purchases = {}
    for (uid, date), group in buys_df[
        buys_df.set_index(["user_id", "date"]).index.isin(single_buy_sessions)
    ].groupby(["user_id", "date"]):
        session_purchases[(uid, date)] = group["item_id"].iloc[0]

    # Build menus from views
    session_menus = {}
    for (uid, date), group in views_df[
        views_df.set_index(["user_id", "date"]).index.isin(single_buy_sessions)
    ].groupby(["user_id", "date"]):
        session_menus[(uid, date)] = set(group["item_id"].tolist())

    del views_df, buys_df

    # Build (menu, choice) records
    records = []
    for key, menu in session_menus.items():
        choice = session_purchases.get(key)
        if choice is None:
            continue
        menu = menu | {choice}
        if len(menu) < MIN_MENU_SIZE or len(menu) > MAX_MENU_SIZE:
            continue
        records.append({"user_id": key[0], "menu": frozenset(menu), "choice": choice})

    print(f"  Valid sessions: {len(records):,}")

    # Group by user
    from collections import defaultdict
    user_sessions = defaultdict(list)
    for r in records:
        user_sessions[r["user_id"]].append(r)

    # Filter by min_sessions and cap at max_users
    qualifying = {uid: sessions for uid, sessions in user_sessions.items()
                  if len(sessions) >= min_sessions}
    # Sort by session count descending (most active users first)
    qualifying = dict(sorted(qualifying.items(), key=lambda x: len(x[1]), reverse=True))

    if max_users is not None:
        qualifying = dict(list(qualifying.items())[:max_users])

    print(f"  Users with >= {min_sessions} sessions: {len(qualifying):,}")

    # Build MenuChoiceLog per user
    user_logs: dict[str, MenuChoiceLog] = {}
    for uid, sessions in qualifying.items():
        menus = [s["menu"] for s in sessions]
        choices = [s["choice"] for s in sessions]

        if remap_items:
            all_items = set()
            for m in menus:
                all_items |= m
            item_map = {item: idx for idx, item in enumerate(sorted(all_items))}
            menus = [frozenset(item_map[i] for i in m) for m in menus]
            choices = [item_map[c] for c in choices]

        user_logs[str(uid)] = MenuChoiceLog(menus=menus, choices=choices)

    print(f"  Built {len(user_logs)} MenuChoiceLog objects")
    return user_logs
