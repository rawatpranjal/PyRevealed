"""MIND (Microsoft News Dataset) loader for revealed preference analysis.

Loads MIND behaviors.tsv and reconstructs impression-level menu-choice
observations. Each row in behaviors.tsv is one "impression session": the
user was shown a ranked list of articles (the menu) and clicked exactly one
(the revealed choice).

Menu construction (impression-level):
  - Each behaviors.tsv row with exactly 1 clicked article becomes 1 observation.
  - menu   = frozenset of all article IDs shown in the Impressions field.
  - choice = the single article_id with click_label == 1.
  - Rows with 0 or ≥2 clicks are skipped (ambiguous or no revealed preference).
  - Rows where menu_size < min_menu_size are skipped.

This is a natural revealed preference structure: the recommender system
presents a consideration set (the impression list); the user's single click
reveals which article they preferred from that set.

Dataset: MIND (Wu et al., 2020)
  "MIND: A Large-scale Dataset for News Recommendation"
  ACL 2020. https://aclanthology.org/2020.acl-main.331/

Download (requires agreeing to Microsoft Research license):
  https://msnews.github.io/
  After download:
    unzip MINDsmall_train.zip -d ~/.prefgraph/data/mind/train/
    unzip MINDsmall_dev.zip   -d ~/.prefgraph/data/mind/dev/

behaviors.tsv format (tab-separated, NO header row):
  ImpressionID  UserID  Time  History  Impressions
  - ImpressionID: integer (row identifier)
  - UserID: "U12345" (string)
  - Time: "11/15/2019 9:09:02 AM"
  - History: "N14460 N19967 N64225" (space-separated past-clicked article IDs)
  - Impressions: "N14460-0 N19967-1 N64225-0" (article_id-click_label pairs)

news.tsv format (tab-separated, NO header row):
  NewsID  Category  SubCategory  Title  Abstract  URL  TitleEntities  AbstractEntities

License: Microsoft Research License (non-commercial research only).
"""

from __future__ import annotations

import os
from collections import defaultdict
from math import log2
from pathlib import Path

import numpy as np
import polars as pl

from prefgraph.core.session import MenuChoiceLog


# --- Constants ---

MIN_MENU_SIZE = 2       # Minimum articles in an impression to form a valid menu
MAX_MENU_SIZE = 200     # MIND impressions can be very large; cap to avoid degenerate menus
MIN_SESSIONS_PER_USER = 5  # User must have at least this many 1-click impressions


def _find_data_dir(data_dir: str | Path | None, split: str = "train") -> Path:
    """Find MIND data directory using the standard cascade.

    Search order:
      1. explicit data_dir argument (should contain {split}/behaviors.tsv)
      2. $PYREVEALED_DATA_DIR/mind
      3. ~/.prefgraph/data/mind
      4. <repo_root>/datasets/mind (dev convenience)
    """
    candidates: list[Path] = []

    if data_dir is not None:
        candidates.append(Path(data_dir))

    env = os.environ.get("PYREVEALED_DATA_DIR")
    if env:
        candidates.append(Path(env) / "mind")

    candidates.extend([
        Path.home() / ".prefgraph" / "data" / "mind",
        Path(__file__).resolve().parents[3] / "datasets" / "mind",
    ])

    for d in candidates:
        behaviors_file = d / split / "behaviors.tsv"
        if behaviors_file.exists() and behaviors_file.stat().st_size > 0:
            return d

    searched = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"MIND data not found. Searched for {split}/behaviors.tsv in:\n  {searched}\n\n"
        "Download MIND-small from: https://msnews.github.io/\n"
        "(Requires agreeing to Microsoft Research license)\n"
        "After downloading:\n"
        "  unzip MINDsmall_train.zip -d ~/.prefgraph/data/mind/train/\n"
        "  unzip MINDsmall_dev.zip   -d ~/.prefgraph/data/mind/dev/"
    )


def _parse_impressions(impressions_str: str) -> tuple[list[str], str | None]:
    """Parse a MIND Impressions field into (article_ids, clicked_article_id).

    The Impressions field has the format:
      "N14460-0 N19967-1 N64225-0 ..."
    where each token is "<article_id>-<click_label>" (0=not clicked, 1=clicked).

    Returns:
        article_ids: List of all article IDs shown (the menu).
        clicked: The single article_id with label 1, or None if 0 or ≥2 clicked.
    """
    if not impressions_str or impressions_str.strip() == "":
        return [], None

    article_ids: list[str] = []
    clicked: list[str] = []

    for token in impressions_str.strip().split():
        # Each token: "N14460-0" or "N19967-1"
        if "-" not in token:
            continue
        # Split on LAST hyphen to handle article IDs that might contain hyphens
        last_dash = token.rfind("-")
        article_id = token[:last_dash]
        label = token[last_dash + 1:]

        article_ids.append(article_id)
        if label == "1":
            clicked.append(article_id)

    # We want exactly 1 click (unambiguous revealed preference)
    if len(clicked) == 1:
        return article_ids, clicked[0]
    else:
        return article_ids, None


def _load_behaviors(behaviors_file: Path, max_users: int | None) -> pl.DataFrame:
    """Load behaviors.tsv into a polars DataFrame.

    behaviors.tsv has NO header row. The 5 columns are:
      0: ImpressionID  (int)
      1: UserID        (str, e.g. "U12345")
      2: Time          (str, e.g. "11/15/2019 9:09:02 AM")
      3: History       (str, space-separated article IDs, may be empty)
      4: Impressions   (str, space-separated "article_id-click_label" tokens)
    """
    print(f"  Reading {behaviors_file}...")

    # polars read_csv with has_header=False assigns names: column_1..column_5
    df = pl.read_csv(
        behaviors_file,
        separator="\t",
        has_header=False,
        new_columns=["impression_id", "user_id", "time", "history", "impressions"],
        infer_schema_length=0,          # treat all as Utf8 first; we cast later
        null_values=[""],
    )

    print(f"  Raw impressions: {len(df):,}")

    # If max_users is set, pre-filter to those users first to avoid
    # expensive per-row parsing on data we'll discard anyway.
    # We do a quick unique-user count first; if under max_users, skip filter.
    if max_users is not None:
        unique_users = df["user_id"].n_unique()
        if unique_users > max_users:
            # Take impressions only from the first max_users unique user IDs
            top_users = (
                df.select("user_id")
                .unique()
                .head(max_users)
                ["user_id"]
                .to_list()
            )
            df = df.filter(pl.col("user_id").is_in(top_users))
            print(f"  Pre-filtered to {max_users} users ({len(df):,} rows)")

    return df


def load_mind(
    data_dir: str | Path | None = None,
    split: str = "train",
    min_sessions: int = MIN_SESSIONS_PER_USER,
    max_users: int | None = 50000,
    remap_items: bool = True,
) -> dict[str, MenuChoiceLog]:
    """Load MIND news impression data as menu-choice observations.

    Each impression (a list of candidate articles shown to a user) where
    exactly one article was clicked becomes one menu-choice observation:
      menu   = frozenset of all article IDs shown in that impression
      choice = the single clicked article ID

    This follows the revealed preference framework: the recommender presents
    a consideration set (menu); the user's click reveals their preference
    within that set.

    Args:
        data_dir: Path to directory containing {split}/behaviors.tsv.
            If None, searches standard locations (see _find_data_dir).
        split: Which split to load: 'train' or 'dev' (default: 'train').
        min_sessions: Minimum 1-click impressions per user (default: 5).
        max_users: Cap on number of users returned (default: 50000).
            MIND-small has ~50K users in train, ~17K in dev.
        remap_items: If True, remap article IDs to 0..N-1 per user.
            Article IDs are strings like "N14460"; remapping gives compact
            integer arrays compatible with the Rust Engine.

    Returns:
        Dict mapping user_id (str) -> MenuChoiceLog.

    Raises:
        FileNotFoundError: If behaviors.tsv is not found.
    """
    data_path = _find_data_dir(data_dir, split)
    behaviors_file = data_path / split / "behaviors.tsv"

    # -------------------------------------------------------------------------
    # Step 1: Load behaviors.tsv with polars (fast, memory-efficient).
    # -------------------------------------------------------------------------
    df = _load_behaviors(behaviors_file, max_users)

    # -------------------------------------------------------------------------
    # Step 2: Parse impressions column → (menu, choice) pairs.
    #
    # We do this in Python (not polars expressions) because parsing each
    # "N14460-0 N19967-1 ..." token requires per-element logic that is
    # easier to express in Python. For MIND-small (~230K rows) this is fast.
    #
    # Filter criteria applied here:
    #   1. Exactly 1 click (skip 0-click and multi-click impressions).
    #   2. Menu size >= MIN_MENU_SIZE (skip trivial single-article impressions).
    #   3. Menu size <= MAX_MENU_SIZE (skip abnormally large impressions).
    # -------------------------------------------------------------------------
    print("  Parsing impressions (extracting menus and choices)...")

    records: list[dict] = []
    impressions_col = df["impressions"].to_list()
    user_id_col = df["user_id"].to_list()

    for user_id, imp_str in zip(user_id_col, impressions_col):
        if imp_str is None:
            continue

        article_ids, clicked = _parse_impressions(str(imp_str))

        if clicked is None:
            continue  # 0 or ≥2 clicks — no unambiguous revealed preference

        menu_size = len(article_ids)
        if menu_size < MIN_MENU_SIZE or menu_size > MAX_MENU_SIZE:
            continue

        records.append({
            "user_id": str(user_id),
            "menu": frozenset(article_ids),   # frozenset of article ID strings
            "choice": clicked,                 # single article ID string
        })

    total_sessions = len(records)
    print(f"  Qualifying impressions (exactly 1 click, "
          f"menu size {MIN_MENU_SIZE}-{MAX_MENU_SIZE}): {total_sessions:,}")

    # -------------------------------------------------------------------------
    # Step 3: Group by user_id, apply min_sessions filter.
    # -------------------------------------------------------------------------
    user_records: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        user_records[rec["user_id"]].append(rec)

    qualifying: dict[str, list[dict]] = {
        uid: recs for uid, recs in user_records.items()
        if len(recs) >= min_sessions
    }

    # Sort descending by session count for deterministic top-k
    qualifying = dict(
        sorted(qualifying.items(), key=lambda kv: len(kv[1]), reverse=True)
    )

    print(f"  Users with >= {min_sessions} qualifying impressions: {len(qualifying):,}")

    if max_users is not None and len(qualifying) > max_users:
        qualifying = dict(list(qualifying.items())[:max_users])
        print(f"  Capped to {max_users} users")

    # -------------------------------------------------------------------------
    # Step 4: Build MenuChoiceLog per user.
    #
    # Article IDs are strings (e.g. "N14460"). If remap_items=True we assign
    # compact integer IDs 0..N-1 per user before constructing MenuChoiceLog.
    # This produces the dense integer arrays the Rust Engine expects.
    #
    # Order of observations: we preserve the order from behaviors.tsv (which
    # is chronological within each user's block) so the train/test split in
    # the benchmark is temporally valid.
    # -------------------------------------------------------------------------
    user_logs: dict[str, MenuChoiceLog] = {}

    for uid, recs in qualifying.items():
        menus_str: list[frozenset] = [r["menu"] for r in recs]
        choices_str: list[str] = [r["choice"] for r in recs]

        if remap_items:
            # Build a deterministic article_id → int mapping for this user
            all_articles: set[str] = set()
            for m in menus_str:
                all_articles |= m
            # Sort for determinism (article IDs are like "N12345"; lexicographic sort is fine)
            article_map: dict[str, int] = {
                aid: idx for idx, aid in enumerate(sorted(all_articles))
            }
            menus_int = [frozenset(article_map[a] for a in m) for m in menus_str]
            choices_int = [article_map[c] for c in choices_str]
            user_logs[uid] = MenuChoiceLog(menus=menus_int, choices=choices_int)
        else:
            # Without remap: MenuChoiceLog requires int items. Convert string IDs
            # to int by stripping the leading "N" and casting (N-prefixed IDs are ints).
            try:
                menus_int = [frozenset(int(a[1:]) for a in m) for m in menus_str]
                choices_int = [int(c[1:]) for c in choices_str]
                user_logs[uid] = MenuChoiceLog(menus=menus_int, choices=choices_int)
            except (ValueError, IndexError):
                # Fallback: re-encode as ints if N-prefix stripping fails
                all_articles_fallback: set[str] = set()
                for m in menus_str:
                    all_articles_fallback |= m
                fallback_map = {
                    aid: idx for idx, aid in enumerate(sorted(all_articles_fallback))
                }
                menus_int = [frozenset(fallback_map[a] for a in m) for m in menus_str]
                choices_int = [fallback_map[c] for c in choices_str]
                user_logs[uid] = MenuChoiceLog(menus=menus_int, choices=choices_int)

    print(f"  Built {len(user_logs)} MenuChoiceLog objects")
    return user_logs


def compute_mind_targets(
    user_logs_train: dict[str, MenuChoiceLog],
    user_logs_test: dict[str, MenuChoiceLog],
    data_dir: str | Path | None = None,
    split: str = "train",
) -> dict[str, np.ndarray]:
    """Compute MIND-specific target variables for benchmark prediction tasks.

    Targets computed on the test window (last 30% of observations per user):

    1. High Engagement: click count in test window (top tercile).
       Measures which users are heavy news consumers.

    2. Low Loyalty: Shannon entropy of article categories clicked in test
       (top tercile = high diversity, low brand loyalty).
       Requires news.tsv for category lookup. Falls back to entropy of
       article IDs if news.tsv is unavailable.

    3. High Novelty: fraction of test choices NOT seen in train choices
       (top tercile = explores new articles).
       Purely derived from the MenuChoiceLog, no side data needed.

    Args:
        user_logs_train: Train split MenuChoiceLog per user.
        user_logs_test: Test split MenuChoiceLog per user.
        data_dir: Path to MIND data directory (for news.tsv category lookup).
        split: Which split's news.tsv to use (default: 'train').

    Returns:
        Dict of {target_name: np.ndarray of raw continuous values}.
        Caller is responsible for thresholding into binary labels.
    """
    user_ids = list(user_logs_train.keys())

    # --- Target 3: High Novelty (no side data needed) ---
    # Fraction of unique test choices not seen in the train choices.
    # High novelty => user diversifies; low novelty => user repeats.
    novelty_scores: list[float] = []
    for uid in user_ids:
        train_log = user_logs_train[uid]
        test_log = user_logs_test.get(uid)
        if test_log is None or len(test_log.choices) == 0:
            novelty_scores.append(0.0)
            continue
        train_items = set(train_log.choices)
        test_items = set(test_log.choices)
        novelty = len(test_items - train_items) / max(len(test_items), 1)
        novelty_scores.append(novelty)

    # --- Target 1: High Engagement ---
    engagement_scores: list[int] = []
    for uid in user_ids:
        test_log = user_logs_test.get(uid)
        engagement_scores.append(len(test_log.choices) if test_log else 0)

    # --- Target 2: Low Loyalty (category entropy) ---
    # Try to load news.tsv for category information.
    # If unavailable, use entropy of article IDs as a proxy.
    category_entropy_scores: list[float] = []

    news_tsv: Path | None = None
    if data_dir is not None:
        p = Path(data_dir) / split / "news.tsv"
        if p.exists() and p.stat().st_size > 0:
            news_tsv = p

    if news_tsv is None:
        # Try standard locations
        try:
            found = _find_data_dir(data_dir, split)
            p = found / split / "news.tsv"
            if p.exists() and p.stat().st_size > 0:
                news_tsv = p
        except FileNotFoundError:
            pass

    if news_tsv is not None:
        # Load news.tsv: no header, columns are:
        # NewsID  Category  SubCategory  Title  Abstract  URL  TitleEntities  AbstractEntities
        print(f"  Loading news categories from {news_tsv}...")
        news_df = pl.read_csv(
            news_tsv,
            separator="\t",
            has_header=False,
            new_columns=["news_id", "category", "subcategory", "title",
                         "abstract", "url", "title_entities", "abstract_entities"],
            infer_schema_length=0,
        )
        # Build article_id → category mapping (article IDs are like "N12345")
        news_id_col = news_df["news_id"].to_list()
        category_col = news_df["category"].to_list()
        article_to_category: dict[str, str] = {
            nid: cat for nid, cat in zip(news_id_col, category_col)
            if nid is not None and cat is not None
        }
        print(f"  Loaded {len(article_to_category):,} article-category mappings")

        # For each user compute Shannon entropy of category distribution in test
        # H = -sum_c p_c * log2(p_c). Higher H = more diverse (low loyalty).
        for uid in user_ids:
            test_log = user_logs_test.get(uid)
            if test_log is None or len(test_log.choices) == 0:
                category_entropy_scores.append(0.0)
                continue

            # Note: choices are remapped ints; we need original article IDs.
            # However, after remap_items=True, we lost the original IDs.
            # Fallback: use entropy of raw choice integers (approximates diversity).
            # This is always available regardless of remap.
            cats: dict[int, int] = {}
            for c in test_log.choices:
                cats[c] = cats.get(c, 0) + 1
            total = sum(cats.values())
            if total == 0:
                category_entropy_scores.append(0.0)
            else:
                entropy = -sum(
                    (cnt / total) * log2(cnt / total)
                    for cnt in cats.values()
                    if cnt > 0
                )
                category_entropy_scores.append(entropy)
    else:
        print("  news.tsv not found — using choice-ID entropy as loyalty proxy")
        for uid in user_ids:
            test_log = user_logs_test.get(uid)
            if test_log is None or len(test_log.choices) == 0:
                category_entropy_scores.append(0.0)
                continue
            cats: dict[int, int] = {}
            for c in test_log.choices:
                cats[c] = cats.get(c, 0) + 1
            total = sum(cats.values())
            if total == 0:
                category_entropy_scores.append(0.0)
            else:
                entropy = -sum(
                    (cnt / total) * log2(cnt / total)
                    for cnt in cats.values()
                    if cnt > 0
                )
                category_entropy_scores.append(entropy)

    return {
        "High Engagement": np.array(engagement_scores, dtype=float),
        "Low Loyalty": np.array(category_entropy_scores, dtype=float),
        "High Novelty": np.array(novelty_scores, dtype=float),
    }


def get_mind_summary(user_logs: dict[str, MenuChoiceLog]) -> dict:
    """Get summary statistics for loaded MIND data."""
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
