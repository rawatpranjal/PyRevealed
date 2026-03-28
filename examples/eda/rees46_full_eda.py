#!/usr/bin/env python3
"""
REES46 Full EDA - Polars-native, full 8,832-user analysis.

Raw data: 2019-Oct.csv (5.3 GB) + 2019-Nov.csv (8.4 GB).
Choice model: server_session → single-purchase sessions only.
Menu = all view events in session ∪ {purchased item}.
Unviewed items are invisible: perfect observability.

Phases:
  0. Raw data profiling (event counts, user/session counts per file)
  1. Session construction funnel (dropout at each filter stage)
  2. User-level distributions (sessions, menu sizes, item diversity)
  3. Revealed preference analysis (SARP, HM, WARP via Engine)
  4. Summary verdict
"""

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from prefgraph import Engine, MenuChoiceLog  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path.home() / ".prefgraph" / "data" / "rees46"
MIN_MENU_SIZE = 2
MAX_MENU_SIZE = 50
MIN_SESSIONS_PER_USER = 5

# Only these 4 columns are needed; skip category, brand, price to save I/O
_COLS = ["event_type", "product_id", "user_id", "user_session"]


# ── Helper ────────────────────────────────────────────────────────────────────

def sec(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


# ── Phase 0 ───────────────────────────────────────────────────────────────────

def phase0_raw_stats(csv_files: list[Path]) -> None:
    """Quick raw-data profiling with Polars lazy scanning.

    Reads only the event_type column for event-type distribution and
    scans product_id/user_id/user_session for cardinality.
    Polars scan_csv predicate-pushes the column selection, so it reads
    ~4 columns instead of the full ~10-column CSV.
    """
    print("=" * 90)
    print("PHASE 0: RAW DATA PROFILING")
    print("=" * 90)

    total_rows = 0
    for f in csv_files:
        t0 = time.time()
        print(f"\nFile: {f.name}  ({f.stat().st_size / 1e9:.1f} GB)")

        # Event-type distribution - collect only event_type column
        event_dist = (
            pl.scan_csv(f, infer_schema_length=10_000)
            .select("event_type")
            .group_by("event_type")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .collect()
        )
        n_rows = event_dist["n"].sum()
        total_rows += n_rows
        print(f"  Total events: {n_rows:>12,}   ({sec(t0)})")
        print("  Event type distribution:")
        for row in event_dist.iter_rows(named=True):
            pct = 100 * row["n"] / n_rows
            print(f"    {row['event_type']:<12}  {row['n']:>10,}  ({pct:>5.1f}%)")

        # Unique users and sessions - collect two columns
        t1 = time.time()
        cardinality = (
            pl.scan_csv(f, infer_schema_length=10_000)
            .select(["user_id", "user_session"])
            .select([
                pl.col("user_id").n_unique().alias("n_users"),
                pl.col("user_session").n_unique().alias("n_sessions"),
            ])
            .collect()
        )
        print(
            f"  Unique users:    {cardinality[0, 'n_users']:>10,}\n"
            f"  Unique sessions: {cardinality[0, 'n_sessions']:>10,}  ({sec(t1)})"
        )

    print(f"\n  TOTAL raw events across both files: {total_rows:,}")


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def phase1_build_sessions(csv_files: list[Path]) -> pl.DataFrame:
    """Build session-level (menu, choice) table with a logged filter funnel.

    Filter cascade:
      raw events
      → view + purchase only
      → sessions with exactly 1 purchase
      → sessions with ≥1 viewed item before purchase
      → menu_size ∈ [MIN_MENU_SIZE, MAX_MENU_SIZE]
      → users with ≥ MIN_SESSIONS_PER_USER sessions

    Returns a DataFrame with columns:
      user_id (i64), user_session (str),
      menu (list[i64]), choice (i64), menu_size (u32)
    """
    print("\n" + "=" * 90)
    print("PHASE 1: SESSION CONSTRUCTION FUNNEL")
    print("=" * 90)

    t0 = time.time()
    print("\nReading and concatenating both CSV files (lazy)...")

    # Lazy scan - Polars reads all needed columns in one pass
    frames = [
        pl.scan_csv(
            f,
            schema_overrides={
                "event_type": pl.Categorical,
                "product_id": pl.Int64,
                "user_id": pl.Int64,
                "user_session": pl.Utf8,
            },
        ).select(_COLS)
        for f in csv_files
    ]
    events_lazy = pl.concat(frames)

    # Filter to view + purchase before any groupby - cuts data ~50%
    events_lazy = events_lazy.filter(
        pl.col("event_type").cast(pl.Utf8).is_in(["view", "purchase"])
    )

    print("  Collecting events (view + purchase only)...")
    events = events_lazy.collect()
    n_vp_events = len(events)
    print(f"  View + purchase events: {n_vp_events:,}   ({sec(t0)})")

    # ── Purchases per session ──────────────────────────────────────────────────
    purchases = events.filter(pl.col("event_type").cast(pl.Utf8) == "purchase")
    purchase_counts = (
        purchases.group_by("user_session")
        .agg([
            pl.len().alias("n_purchases"),
            pl.col("product_id").first().alias("choice"),
            pl.col("user_id").first().alias("user_id"),
        ])
    )
    n_sessions_raw = len(purchase_counts)
    print(f"\n  Sessions with ≥1 purchase:         {n_sessions_raw:>10,}")

    # Keep only single-purchase sessions
    single_purchase = purchase_counts.filter(pl.col("n_purchases") == 1)
    n_single = len(single_purchase)
    print(f"  Sessions with exactly 1 purchase:  {n_single:>10,}  "
          f"({100*n_single/n_sessions_raw:.1f}% of above)")

    # ── Views per session ──────────────────────────────────────────────────────
    views = events.filter(pl.col("event_type").cast(pl.Utf8) == "view")
    views_per_session = (
        views.group_by("user_session")
        .agg(pl.col("product_id").alias("viewed_items"))
    )
    print(f"  Sessions with ≥1 view:             {len(views_per_session):>10,}")

    # Free raw events - no longer needed
    del events, purchases, views

    # ── Join sessions + views ─────────────────────────────────────────────────
    sessions = single_purchase.join(
        views_per_session, on="user_session", how="inner"
    )
    n_inner = len(sessions)
    print(f"  Sessions with 1 purchase + views:  {n_inner:>10,}  "
          f"({100*n_inner/n_single:.1f}% of single-purchase)")

    del single_purchase, views_per_session

    # ── Build menu = viewed ∪ {choice} using Python (row-level) ───────────────
    print("\n  Building menus (viewed_items ∪ {choice})...")
    viewed_lists = sessions["viewed_items"].to_list()
    choices = sessions["choice"].to_list()

    menus: list[list[int]] = []
    for viewed, c in zip(viewed_lists, choices):
        menu_set = set(viewed) | {c}
        menus.append(sorted(menu_set))

    sessions = sessions.with_columns([
        pl.Series("menu", menus, dtype=pl.List(pl.Int64))
    ])
    sessions = sessions.with_columns(
        pl.col("menu").list.len().alias("menu_size")
    )

    # Drop viewed_items and n_purchases - no longer needed
    sessions = sessions.drop(["viewed_items", "n_purchases"])

    # ── Menu size filter ───────────────────────────────────────────────────────
    n_before_size = len(sessions)
    sessions = sessions.filter(
        (pl.col("menu_size") >= MIN_MENU_SIZE) &
        (pl.col("menu_size") <= MAX_MENU_SIZE)
    )
    n_after_size = len(sessions)
    print(f"\n  After menu size filter [{MIN_MENU_SIZE},{MAX_MENU_SIZE}]:")
    print(f"    Before: {n_before_size:>10,}")
    print(f"    After:  {n_after_size:>10,}  ({100*n_after_size/n_before_size:.1f}% retained)")

    # ── User session count filter ──────────────────────────────────────────────
    user_session_counts = sessions.group_by("user_id").agg(pl.len().alias("n_sessions"))
    qualifying_users = user_session_counts.filter(
        pl.col("n_sessions") >= MIN_SESSIONS_PER_USER
    )["user_id"]

    sessions = sessions.filter(pl.col("user_id").is_in(qualifying_users))
    n_users = sessions["user_id"].n_unique()
    n_sessions_final = len(sessions)
    print(f"\n  After ≥{MIN_SESSIONS_PER_USER} sessions/user filter:")
    print(f"    Users:    {n_users:>10,}")
    print(f"    Sessions: {n_sessions_final:>10,}  ({sec(t0)} total)")

    return sessions


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def phase2_distributions(sessions: pl.DataFrame) -> None:
    """User-level and menu-level distribution statistics."""
    print("\n" + "=" * 90)
    print("PHASE 2: USER AND MENU DISTRIBUTIONS")
    print("=" * 90)

    # Sessions per user
    spuser = (
        sessions.group_by("user_id")
        .agg(pl.len().alias("n_sessions"))
        ["n_sessions"].to_numpy()
    )
    print(f"\nSessions per user (N={len(spuser):,} users):")
    _print_dist(spuser)
    print("  Decile breakdown:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:>3}: {np.percentile(spuser, p):.0f}")

    # Menu sizes
    ms = sessions["menu_size"].to_numpy()
    print(f"\nMenu size distribution (N={len(ms):,} sessions):")
    _print_dist(ms)

    # Top 15 menu sizes
    mc = {}
    for v in ms:
        mc[int(v)] = mc.get(int(v), 0) + 1
    print("\n  Top 15 most common menu sizes:")
    for size in sorted(mc, key=mc.get, reverse=True)[:15]:
        pct = 100 * mc[size] / len(ms)
        bar = "█" * int(pct / 2)
        print(f"    size {size:>3}: {mc[size]:>9,} sessions ({pct:>5.1f}%)  {bar}")

    # Item diversity per user
    print("\nItem diversity per user (unique items ever seen in menus):")
    item_counts = []
    for uid, grp in sessions.group_by("user_id"):
        all_items: set[int] = set()
        for menu in grp["menu"].to_list():
            all_items.update(menu)
        item_counts.append(len(all_items))
    ic = np.array(item_counts)
    _print_dist(ic)

    # Repeat item rate (fraction of sessions where choice was in a previous menu)
    print("\nSession-level choice coverage (was choice ever seen before in prior session?):")
    _analyze_repeat_choices(sessions)


def _print_dist(arr: np.ndarray) -> None:
    print(
        f"  Min={arr.min():.1f}  Q25={np.percentile(arr,25):.1f}  "
        f"Median={np.percentile(arr,50):.1f}  Mean={arr.mean():.2f}  "
        f"Q75={np.percentile(arr,75):.1f}  Max={arr.max():.1f}"
    )


def _analyze_repeat_choices(sessions: pl.DataFrame) -> None:
    """For each user, check: what fraction of choices were also in a prior menu?"""
    seen_before_rates = []
    for uid, grp in sessions.group_by("user_id"):
        menus = grp.sort("user_session")["menu"].to_list()
        choices = grp.sort("user_session")["choice"].to_list()
        if len(menus) < 2:
            continue
        cumulative_seen: set[int] = set()
        seen_before_count = 0
        for i, (menu, choice) in enumerate(zip(menus, choices)):
            if choice in cumulative_seen:
                seen_before_count += 1
            cumulative_seen.update(menu)
        rate = seen_before_count / len(choices)
        seen_before_rates.append(rate)
    arr = np.array(seen_before_rates)
    print(f"  (Fraction of choices that appeared in any prior session's menu)")
    _print_dist(arr)
    print(f"  {(arr > 0).mean():.1%} of users have ≥1 cross-session repeat choice")


# ── Phase 3 ───────────────────────────────────────────────────────────────────

def phase3_rp_analysis(sessions: pl.DataFrame) -> None:
    """Full RP analysis using Engine.analyze_menus on all qualifying users.

    Per SARP / WARP / Houtman-Maks (Richter 1966; Houtman & Maks 1985).
    Engine processes all users in a single batch via Rust/Rayon parallelism.
    """
    print("\n" + "=" * 90)
    print("PHASE 3: REVEALED PREFERENCE ANALYSIS (Engine.analyze_menus)")
    print("=" * 90)

    t0 = time.time()

    # Build per-user MenuChoiceLog objects
    # Remap item IDs to 0..N-1 per user (compact integer arrays for Rust)
    print("\nBuilding per-user MenuChoiceLog objects (with local item remapping)...")

    engine_tuples: list[tuple[list[tuple[int, ...]], list[int]]] = []
    user_ids: list[int] = []

    for uid, grp in sessions.group_by("user_id"):
        menus_raw = grp["menu"].to_list()  # list of list[int]
        choices_raw = grp["choice"].to_list()  # list[int]

        # Remap items to 0..N-1 for compact Engine input
        all_items: set[int] = set()
        for m in menus_raw:
            all_items.update(m)
        item_map = {item: idx for idx, item in enumerate(sorted(all_items))}

        menus_remapped = [tuple(sorted(item_map[i] for i in m)) for m in menus_raw]
        choices_remapped = [item_map[c] for c in choices_raw]

        engine_tuples.append((menus_remapped, choices_remapped))
        user_ids.append(uid)

    n_users = len(engine_tuples)
    print(f"  Prepared {n_users:,} users for batch analysis   ({sec(t0)})")

    # Run Engine batch analysis
    print(f"\nRunning Engine.analyze_menus() on {n_users:,} users...")
    engine = Engine()
    t1 = time.time()
    try:
        results = engine.analyze_menus(menus_choices=engine_tuples)
        print(f"  Engine completed: {len(results):,} result objects   ({sec(t1)})")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    # ── Extract metrics ────────────────────────────────────────────────────────
    sarp_consistent = []
    warp_consistent = []
    hm_scores = []

    for r in results:
        if hasattr(r, "sarp_consistent"):
            sarp_consistent.append(int(r.sarp_consistent))
        if hasattr(r, "warp_consistent"):
            warp_consistent.append(int(r.warp_consistent))
        # HM score: fraction of observations that are consistent
        if hasattr(r, "hm") and r.hm is not None:
            if isinstance(r.hm, dict) and "score" in r.hm:
                hm_scores.append(float(r.hm["score"]))
            elif isinstance(r.hm, (int, float)):
                hm_scores.append(float(r.hm))

    n_analyzed = len(sarp_consistent)

    # SARP
    n_sarp_ok = sum(sarp_consistent)
    print(f"\nSARP consistency (N={n_analyzed:,} users):")
    print(f"  Consistent:   {n_sarp_ok:>7,}  ({100*n_sarp_ok/max(n_analyzed,1):.1f}%)")
    print(f"  Violations:   {n_analyzed - n_sarp_ok:>7,}  ({100*(n_analyzed-n_sarp_ok)/max(n_analyzed,1):.1f}%)")

    # WARP
    if warp_consistent:
        n_warp_ok = sum(warp_consistent)
        print(f"\nWARP consistency:")
        print(f"  Consistent:   {n_warp_ok:>7,}  ({100*n_warp_ok/n_analyzed:.1f}%)")
        print(f"  Violations:   {n_analyzed - n_warp_ok:>7,}  ({100*(n_analyzed-n_warp_ok)/n_analyzed:.1f}%)")

    # HM score
    if hm_scores:
        hm = np.array(hm_scores)
        print(f"\nHoutman-Maks score (fraction of observations consistent; N={len(hm):,}):")
        _print_dist(hm)
        print("  Percentile buckets:")
        for lo, hi in [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0), (1.0, 1.01)]:
            hi_label = "1.0" if hi >= 1.0 else f"{hi:.1f}"
            count = np.sum((hm >= lo) & (hm < hi))
            print(f"    HM ∈ [{lo:.1f}, {hi_label}):  {count:>6,}  ({100*count/len(hm):.1f}%)")

    # Sessions × SARP cross-tab (are users with more sessions more/less consistent?)
    if sarp_consistent and n_analyzed > 0:
        print("\nSARP consistency vs. session count (do longer histories matter?):")
        session_counts = (
            sessions.group_by("user_id")
            .agg(pl.len().alias("n_sessions"))
            .sort("user_id")["n_sessions"]
            .to_list()
        )
        # Zip by order (both sorted by groupby order)
        pairs = list(zip(sarp_consistent, session_counts[:n_analyzed]))
        for lo, hi in [(5, 10), (10, 20), (20, 50), (50, 10000)]:
            subset = [s for s, n in pairs if lo <= n < hi]
            if subset:
                pct = 100 * sum(subset) / len(subset)
                print(f"    {lo:>3}–{hi:<5} sessions: n={len(subset):>5,}  "
                      f"SARP-consistent={pct:.1f}%")

    print(f"\n  Total RP analysis time: {sec(t0)}")


# ── Phase 4: Summary ──────────────────────────────────────────────────────────

def phase4_summary(sessions: pl.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("PHASE 4: BENCHMARK SUMMARY")
    print("=" * 90)

    n_users = sessions["user_id"].n_unique()
    n_sessions = len(sessions)
    ms = sessions["menu_size"].to_numpy()

    print(f"""
Dataset: REES46 E-Commerce (Oct–Nov 2019)
  Users:          {n_users:>8,}
  Sessions:       {n_sessions:>8,}
  Median menu:    {np.median(ms):>8.1f} items
  Mean menu:      {ms.mean():>8.2f} items

Assumptions locked in:
  1. Session boundary: user_session column (server-defined, gold standard)
  2. Menu = views in session ∪ {{purchased item}}  (perfect observability)
  3. Single-purchase sessions only (clear choice identification)
  4. Menu size ∈ [{MIN_MENU_SIZE}, {MAX_MENU_SIZE}] (quality filter)
  5. ≥{MIN_SESSIONS_PER_USER} sessions per user (sufficient preference signal)
  6. Item IDs remapped to 0..N-1 per user (compact representation)

Benchmark position:
  ✓ Largest dataset: 8k+ users, real e-commerce behavior
  ✓ Gold-standard observability (server session IDs)
  ✓ No prices / no budget dimension → ordinal preferences only
  ✓ Engine API: batch SARP/HM/WARP via Rust/Rayon

Caveats:
  • Ordinal only (no price variation, no CCEI/MPI)
  • Short horizons per user (few sessions)
  • Recommendation system may filter menus non-randomly
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    csv_files = sorted(DATA_DIR.glob("2019-*.csv"))
    if not csv_files:
        print(f"ERROR: No 2019-*.csv files in {DATA_DIR}")
        print("Download: kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store")
        sys.exit(1)

    print("=" * 90)
    print("REES46 E-COMMERCE FULL EDA  (Polars-native, all users)")
    print(f"Files: {[f.name for f in csv_files]}")
    print("=" * 90)
    t_global = time.time()

    phase0_raw_stats(csv_files)
    sessions = phase1_build_sessions(csv_files)
    phase2_distributions(sessions)
    phase3_rp_analysis(sessions)
    phase4_summary(sessions)

    print(f"Total wall-clock time: {time.time() - t_global:.1f}s")


if __name__ == "__main__":
    main()
