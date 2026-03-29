"""FINN.no Recsys Slates dataset loader.

Loads the FINN.no Slate Dataset (Eide et al., 2021) — a large-scale
sequential recommendation dataset from Norway's largest classifieds marketplace.
Each interaction records the full slate (all items shown to the user) and
which item was clicked, making this dataset ideal for PrefGraph menu-choice analysis.

Unlike session-reconstructed menus (Taobao, REES46), this dataset records the
*directly observed* choice set — the platform explicitly logged what was shown.
This is the gold standard for WARP/SARP testing: the menu is not inferred.

Data format (data.npz keys):
  click : int32[N, T]      — clicked item ID at each time step (0 = no click)
  slate : int32[N, T, K]   — item IDs shown at each time step (0/1/2 = padding)

Special item IDs 0, 1, 2 are padding tokens, not real items. Real items start at 3.
N ≈ 2,300,000 users, T = 20 time steps, K = 25 slate slots.

Data is stored in a single .npz file (~3GB for int32 version).

Download instructions:
  Option A: via recsys_slates_dataset package
    pip install recsys_slates_dataset gdown
    python3 -c "
    from recsys_slates_dataset.data_helper import download_data_files
    download_data_files(data_dir='/Volumes/Expansion/datasets/finn_slates')
    "

  Option B: Manual browser download from Google Drive
    data.npz    : https://drive.google.com/uc?id=1XHqyk01qi9qnvBTfWWwqgDzrdjv1eBVV
    ind2val.json: https://drive.google.com/uc?id=1WOCKfuttMacCb84yQYcRjxjEtgPp6F4N
    itemattr.npz: https://drive.google.com/uc?id=1rKKyMQZqWp8vQ-Pl1SeHrQxzc5dXldnR
    Place all three files in ~/.prefgraph/data/finn_slates/ or $PYREVEALED_DATA_DIR/finn_slates/

Reference:
  Eide, S.O., Zhang, Y., Su, X., Balog, K., and Jain, A. (2021).
  "Large-scale Interactive Recommendation with Tree-Structured Policy Gradient."
  RecSys 2021. Dataset: https://github.com/finn-no/recsys_slates_dataset
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from prefgraph.core.session import MenuChoiceLog


# --- Constants ---

MIN_MENU_SIZE = 2    # Minimum real items in a slate after filtering padding
MAX_MENU_SIZE = 25   # FINN.no slates have K=25 slots, all potentially real items
MIN_SESSIONS_PER_USER = 5  # User must have this many click events (excluding no-click)
PADDING_THRESHOLD = 3  # Item IDs 0, 1, 2 are padding/special; real items >= 3


def _find_data_dir(data_dir: str | Path | None) -> Path:
    """Find FINN.no Slates data directory using the standard cascade.

    Search order:
      1. explicit data_dir argument
      2. $PYREVEALED_DATA_DIR/finn_slates
      3. ~/.prefgraph/data/finn_slates
      4. <repo_root>/datasets/finn_slates (dev convenience)
    """
    candidates: list[Path] = []

    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "finn_slates")

    candidates.extend([
        Path.home() / ".prefgraph" / "data" / "finn_slates",
        Path(__file__).resolve().parents[3] / "datasets" / "finn_slates",
    ])

    for d in candidates:
        if d.is_dir() and (d / "data.npz").exists():
            npz = d / "data.npz"
            if npz.stat().st_size > 1_000_000:  # Must be > 1MB (not an error page)
                return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"FINN.no Slates data not found. Searched:\n  {searched}\n\n"
        "Download options:\n"
        "  A) pip install recsys_slates_dataset gdown==4.5.1\n"
        "     python3 -c \"from recsys_slates_dataset.data_helper import download_data_files; "
        "download_data_files(data_dir='~/.prefgraph/data/finn_slates')\"\n\n"
        "  B) Manual Google Drive download (browser):\n"
        "     data.npz    → https://drive.google.com/uc?id=1XHqyk01qi9qnvBTfWWwqgDzrdjv1eBVV\n"
        "     ind2val.json → https://drive.google.com/uc?id=1WOCKfuttMacCb84yQYcRjxjEtgPp6F4N\n"
        "     itemattr.npz → https://drive.google.com/uc?id=1rKKyMQZqWp8vQ-Pl1SeHrQxzc5dXldnR\n"
        "     Place all three in ~/.prefgraph/data/finn_slates/ or $PYREVEALED_DATA_DIR/finn_slates/"
    )


def load_finn_slates(
    data_dir: str | Path | None = None,
    min_sessions: int = MIN_SESSIONS_PER_USER,
    max_users: int | None = 100_000,
    remap_items: bool = True,
) -> dict[str, MenuChoiceLog]:
    """Load FINN.no Slates data as observed-slate menu-choice observations.

    Unlike session-reconstructed menus, each interaction here has a
    *directly observed* choice set: the platform explicitly logged all
    items shown on the page (the slate = the menu). The clicked item is
    the revealed preference.

    No-click interactions (click == 0 or < PADDING_THRESHOLD) are dropped
    because MenuChoiceLog requires an explicit choice. The no-click rate is
    typically ~40% in this dataset; retained clicks still yield ~37M observations.

    Args:
        data_dir: Path to directory containing data.npz.
            If None, searches standard locations (see _find_data_dir).
        min_sessions: Minimum click events per user (default: 5).
        max_users: Cap on number of users. Default 100,000 (full dataset: 2.3M).
        remap_items: If True, remap item IDs to 0..N-1 per user.

    Returns:
        Dict mapping user_index (str) -> MenuChoiceLog.

    Raises:
        FileNotFoundError: If data.npz is not found or is too small.
        KeyError: If data.npz is missing expected 'click' or 'slate' arrays.
    """
    data_path = _find_data_dir(data_dir)
    npz_path = data_path / "data.npz"

    print(f"  Loading FINN.no Slates from {npz_path}...")
    print(f"  (This is a large file ~3GB; first load may take 30–60 seconds)")

    # -------------------------------------------------------------------------
    # Step 1: Load numpy arrays from data.npz.
    #
    # Expected shapes:
    #   click : int32[N, T]      — clicked item ID per user per timestep
    #   slate : int32[N, T, K]   — item IDs shown per user per timestep
    #
    # Item IDs 0, 1, 2 are special/padding tokens. Real items start at ID 3.
    # -------------------------------------------------------------------------
    with np.load(npz_path, allow_pickle=False) as data_np:
        keys = list(data_np.keys())
        print(f"  data.npz keys: {keys}")

        if "click" not in keys or "slate" not in keys:
            raise KeyError(
                f"data.npz must contain 'click' and 'slate' arrays. Found: {keys}"
            )

        click = data_np["click"]  # [N, T]
        slate = data_np["slate"]  # [N, T, K]

    n_users, T = click.shape
    K = slate.shape[2] if slate.ndim == 3 else 1
    print(f"  Dataset: {n_users:,} users × {T} timesteps × {K} slate slots")
    print(f"  Click rate (non-zero): {(click >= PADDING_THRESHOLD).mean():.1%}")

    # -------------------------------------------------------------------------
    # Step 2: Cap users if needed.
    #
    # Users are stored row-by-row in numpy arrays. We select the first
    # max_users rows (arbitrary ordering from the original dataset).
    # -------------------------------------------------------------------------
    if max_users is not None and n_users > max_users:
        click = click[:max_users]
        slate = slate[:max_users]
        n_users = max_users
        print(f"  Capped to {max_users:,} users")

    # -------------------------------------------------------------------------
    # Step 3: Extract (menu, choice) pairs per user.
    #
    # For each user i and timestep t:
    #   - Real items in slate: {item for item in slate[i, t] if item >= PADDING_THRESHOLD}
    #   - Valid click: click[i, t] >= PADDING_THRESHOLD
    #   - Menu: frozenset of real slate items
    #   - Choice: click[i, t]
    #
    # We only keep observations where:
    #   1. click is a real item (>= PADDING_THRESHOLD)
    #   2. menu has >= MIN_MENU_SIZE real items
    #   3. choice is in menu (sanity check — should always hold)
    # -------------------------------------------------------------------------
    print(f"  Extracting (menu, choice) pairs from {n_users:,} users × {T} steps...")

    user_logs: dict[str, MenuChoiceLog] = {}
    n_kept = 0
    n_skipped_no_click = 0
    n_skipped_small_menu = 0
    n_skipped_choice_not_in_menu = 0

    for i in range(n_users):
        user_clicks = click[i]     # [T]
        user_slates = slate[i]     # [T, K]

        menus: list[frozenset[int]] = []
        choices: list[int] = []

        for t in range(T):
            c = int(user_clicks[t])

            # Skip no-click or padding click
            if c < PADDING_THRESHOLD:
                n_skipped_no_click += 1
                continue

            # Build menu: all real items in the slate at this timestep
            menu = frozenset(
                int(item) for item in user_slates[t]
                if int(item) >= PADDING_THRESHOLD
            )

            if len(menu) < MIN_MENU_SIZE:
                n_skipped_small_menu += 1
                continue

            # Validate choice is in menu (should always hold; log if not)
            if c not in menu:
                n_skipped_choice_not_in_menu += 1
                # Include the choice in the menu to preserve the observation
                menu = menu | {c}

            menus.append(menu)
            choices.append(c)

        # Only keep users with enough click sessions
        if len(choices) < min_sessions:
            continue

        if remap_items:
            # Remap item IDs to 0..N-1 for compact representation
            all_items: set[int] = set()
            for m in menus:
                all_items |= m
            item_map = {item: idx for idx, item in enumerate(sorted(all_items))}
            menus = [frozenset(item_map[v] for v in m) for m in menus]
            choices = [item_map[c] for c in choices]

        user_logs[str(i)] = MenuChoiceLog(menus=menus, choices=choices)
        n_kept += len(choices)

    print(f"  Users with >= {min_sessions} click sessions: {len(user_logs):,}")
    print(f"  Total kept observations: {n_kept:,}")
    print(f"  Skipped — no-click: {n_skipped_no_click:,}  "
          f"small-menu: {n_skipped_small_menu:,}  "
          f"choice-not-in-slate: {n_skipped_choice_not_in_menu:,}")

    return user_logs
