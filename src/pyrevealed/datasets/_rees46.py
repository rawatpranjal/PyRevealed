"""REES46 eCommerce behavior dataset loader.

Loads REES46 multi-category eCommerce event data and reconstructs
menu-choice observations for revealed preference analysis. Each user
session (items viewed before a purchase) becomes a menu-choice
observation: viewed items = menu, purchased item = choice.

Data must be downloaded separately from Kaggle:
  kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store
  unzip ecommerce-behavior-data-from-multi-category-store.zip -d ~/.pyrevealed/data/rees46/

Source: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
License: CC0 (Public Domain)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pyrevealed.core.session import MenuChoiceLog


# --- Constants ---

MIN_MENU_SIZE = 2  # Minimum items viewed in session (including purchase)
MIN_SESSIONS_PER_USER = 5  # Minimum sessions for a user to be included
MAX_MENU_SIZE = 50  # Cap menu size to avoid degenerate sessions
MAX_USERS = 10_000  # Default cap on number of users returned

# CSV columns we actually need (skip category_id, category_code, brand to save memory)
_USE_COLS = ["event_type", "product_id", "user_id", "user_session"]

# Dtype spec for memory-efficient CSV reading
_DTYPES = {
    "event_type": "category",
    "product_id": "int64",
    "user_id": "int64",
    "user_session": "category",
}


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find REES46 data directory via cascade."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "rees46")

    candidates.extend([
        Path.home() / ".pyrevealed" / "data" / "rees46",
        Path(__file__).resolve().parents[3] / "datasets" / "rees46" / "data",
    ])

    for d in candidates:
        if d.is_dir():
            csvs = list(d.glob("2019-*.csv"))
            if csvs:
                return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"REES46 data not found. Searched:\n  {searched}\n\n"
        "Download from Kaggle:\n"
        "  kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store\n"
        "  unzip ecommerce-behavior-data-from-multi-category-store.zip "
        "-d ~/.pyrevealed/data/rees46/\n\n"
        "Expected CSV files: 2019-Oct.csv, 2019-Nov.csv"
    )


def _load_events(data_path: Path) -> pd.DataFrame:
    """Load and concatenate monthly CSV files with memory-efficient dtypes.

    Only reads the columns needed for menu-choice extraction.
    """
    csv_files = sorted(data_path.glob("2019-*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No 2019-*.csv files found in {data_path}")

    chunks = []
    for f in csv_files:
        print(f"  Reading {f.name}...")
        df = pd.read_csv(
            f,
            usecols=_USE_COLS,
            dtype=_DTYPES,
        )
        chunks.append(df)

    events = pd.concat(chunks, ignore_index=True)
    return events


def _extract_menu_choices(
    events: pd.DataFrame,
    min_menu_size: int = MIN_MENU_SIZE,
    max_menu_size: int = MAX_MENU_SIZE,
) -> pd.DataFrame:
    """Extract (menu, choice) pairs from sessions.

    Menu = set of product_ids viewed in the session.
    Choice = product_id purchased in the session.
    Only keeps sessions with exactly 1 purchase and >= min_menu_size viewed items.
    """
    # Split by event type for vectorized groupby
    views = events.loc[events["event_type"] == "view"]
    purchases = events.loc[events["event_type"] == "purchase"]

    # Count purchases per session — keep only sessions with exactly 1
    purchase_counts = purchases.groupby("user_session").size()
    single_purchase_sessions = purchase_counts[purchase_counts == 1].index

    purchases_filtered = purchases[
        purchases["user_session"].isin(single_purchase_sessions)
    ]

    # Build menu (set of viewed items) per session
    viewed_per_session = (
        views.groupby("user_session")["product_id"]
        .apply(frozenset)
        .rename("viewed")
    )

    # Merge: session -> (viewed set, purchased item, user_id)
    session_df = purchases_filtered[["user_session", "product_id", "user_id"]].copy()
    session_df = session_df.rename(columns={"product_id": "choice"})
    session_df = session_df.merge(
        viewed_per_session, left_on="user_session", right_index=True, how="inner"
    )

    # Menu = viewed items union purchased item
    session_df["menu"] = session_df.apply(
        lambda row: row["viewed"] | frozenset({row["choice"]}), axis=1
    )
    session_df["menu_size"] = session_df["menu"].apply(len)

    # Filter by menu size
    session_df = session_df[
        (session_df["menu_size"] >= min_menu_size)
        & (session_df["menu_size"] <= max_menu_size)
    ]

    return session_df[["user_id", "user_session", "menu", "choice", "menu_size"]]


def load_rees46(
    data_dir: str | Path | None = None,
    min_sessions: int = MIN_SESSIONS_PER_USER,
    max_users: int = MAX_USERS,
    remap_items: bool = True,
) -> dict[str, MenuChoiceLog]:
    """Load REES46 eCommerce behavior data as menu-choice observations.

    Reconstructs menus from session data: items viewed in a session form
    the menu, the purchased item is the choice. Sessions are identified
    by the user_session column in the raw data.

    Args:
        data_dir: Path to directory containing 2019-Oct.csv, 2019-Nov.csv.
            If None, searches standard locations.
        min_sessions: Minimum sessions per user (default: 5).
        max_users: Limit number of users returned (default: 10000).
        remap_items: If True, remap item IDs to 0..N-1 per user for
            compact representation.

    Returns:
        Dict mapping user_id (str) -> MenuChoiceLog.

    Raises:
        FileNotFoundError: If CSV files are not found.
    """
    data_path = _find_data_dir(data_dir)

    print(f"  Loading REES46 events from {data_path}...")
    events = _load_events(data_path)
    print(f"  Raw events: {len(events):,}")

    # Keep only view and purchase events (drop cart — not needed for menus)
    events = events[events["event_type"].isin(["view", "purchase"])]
    print(f"  View + purchase events: {len(events):,}")

    # Extract menu-choice pairs
    print("  Extracting menu-choice pairs...")
    choices_df = _extract_menu_choices(events)
    print(
        f"  Valid sessions (1 purchase, menu size {MIN_MENU_SIZE}-{MAX_MENU_SIZE}): "
        f"{len(choices_df):,}"
    )

    # Free memory — events no longer needed
    del events

    # Filter users with enough sessions
    user_counts = choices_df["user_id"].value_counts()
    qualifying_users = user_counts[user_counts >= min_sessions].index
    choices_df = choices_df[choices_df["user_id"].isin(qualifying_users)]
    print(f"  Users with >= {min_sessions} sessions: {len(qualifying_users):,}")

    # Cap at max_users
    if max_users is not None and len(qualifying_users) > max_users:
        qualifying_users = qualifying_users[:max_users]
        choices_df = choices_df[choices_df["user_id"].isin(qualifying_users)]
        print(f"  Capped to {max_users} users")

    # Build MenuChoiceLog per user
    user_logs: dict[str, MenuChoiceLog] = {}

    for user_id, user_data in choices_df.groupby("user_id"):
        menus = list(user_data["menu"])
        chosen = list(user_data["choice"])

        if remap_items:
            # Remap item IDs to 0..N-1 for compact representation
            all_items: set[int] = set()
            for m in menus:
                all_items |= m
            item_map = {item: idx for idx, item in enumerate(sorted(all_items))}

            menus = [frozenset(item_map[i] for i in m) for m in menus]
            chosen = [item_map[c] for c in chosen]

        log = MenuChoiceLog(menus=menus, choices=chosen)
        user_logs[str(user_id)] = log

    print(f"  Built {len(user_logs)} MenuChoiceLog objects")
    return user_logs


def get_rees46_summary(user_logs: dict[str, MenuChoiceLog]) -> dict:
    """Get summary statistics for loaded REES46 data."""
    if not user_logs:
        return {"n_users": 0}

    sessions_per_user = [len(log.choices) for log in user_logs.values()]
    menu_sizes = []
    for log in user_logs.values():
        menu_sizes.extend(len(m) for m in log.menus)

    return {
        "n_users": len(user_logs),
        "total_sessions": sum(sessions_per_user),
        "mean_sessions": np.mean(sessions_per_user),
        "median_sessions": np.median(sessions_per_user),
        "mean_menu_size": np.mean(menu_sizes),
        "median_menu_size": np.median(menu_sizes),
    }
