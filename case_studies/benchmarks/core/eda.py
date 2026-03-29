"""Standardized EDA summary statistics across all benchmark datasets.

Produces a flat dict of comparable metrics for budget (BehaviorLog) and
menu (MenuChoiceLog) datasets. Metrics are chosen to answer: does this
data look like real consumer choice behavior?
"""

from __future__ import annotations

import json
import re
from collections import Counter
from math import log2
from pathlib import Path

import numpy as np


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ---------------------------------------------------------------------------
# Budget EDA (prices × quantities)
# ---------------------------------------------------------------------------

def compute_budget_eda(
    train_tuples: list[tuple[np.ndarray, np.ndarray]],
    user_ids: list[str],
) -> dict:
    """Standardized EDA for budget datasets.

    Args:
        train_tuples: List of (prices, quantities) arrays per user.
            prices: (T, K) array, quantities: (T, K) array.
        user_ids: Corresponding user identifiers.

    Returns:
        Flat dict of EDA metrics.
    """
    n_users = len(user_ids)
    if n_users == 0:
        return {"n_users": 0}

    obs_per_user = []
    all_spends = []       # per-observation total spend
    all_prices = []       # non-zero prices
    active_k_per_obs = [] # goods purchased per observation
    hhis = []             # Herfindahl per user
    top_shares = []       # max-category share per user
    spend_cvs = []        # CV of spend per user
    zero_cells = 0
    total_cells = 0
    repeat_count = 0
    total_obs_for_repeat = 0
    unique_fracs = []

    K = train_tuples[0][0].shape[1] if train_tuples else 0

    for prices, qty in train_tuples:
        T = prices.shape[0]
        obs_per_user.append(T)

        # Spend per observation
        spend_per_obs = np.sum(prices * qty, axis=1)
        all_spends.extend(spend_per_obs.tolist())

        # Non-zero prices
        nz_mask = prices > 0
        if nz_mask.any():
            all_prices.extend(prices[nz_mask].tolist())

        # Active goods per observation
        for t in range(T):
            active = int(np.sum(qty[t] > 0))
            active_k_per_obs.append(active)

        # Zero-spend rate
        zero_cells += int(np.sum(qty == 0))
        total_cells += qty.size

        # HHI and top-category share (user-level)
        total_per_cat = np.sum(qty, axis=0)
        total_all = np.sum(total_per_cat)
        if total_all > 0:
            shares = total_per_cat / total_all
            hhis.append(float(np.sum(shares ** 2)))
            top_shares.append(float(np.max(shares)))

        # Spend CV
        if len(spend_per_obs) > 1 and np.mean(spend_per_obs) > 0:
            spend_cvs.append(float(np.std(spend_per_obs) / np.mean(spend_per_obs)))

        # Repeat rate: fraction of obs where the "top good" was also top in previous obs
        # Approximate: count obs where argmax(qty) == argmax(qty) of previous obs
        if T > 1:
            top_goods = np.argmax(qty, axis=1)
            repeats = int(np.sum(top_goods[1:] == top_goods[:-1]))
            repeat_count += repeats
            total_obs_for_repeat += T - 1

        # Unique choices fraction (unique top-goods / T)
        if T > 0:
            top_goods = np.argmax(qty, axis=1)
            unique_fracs.append(len(set(top_goods.tolist())) / T)

    obs_arr = np.array(obs_per_user)
    spend_arr = np.array(all_spends)
    price_arr = np.array(all_prices) if all_prices else np.array([0.0])

    return {
        "type": "budget",
        "n_users": n_users,
        "total_obs": int(obs_arr.sum()),
        "T_median": float(np.median(obs_arr)),
        "T_p25": float(np.percentile(obs_arr, 25)),
        "T_p75": float(np.percentile(obs_arr, 75)),
        "T_min": int(obs_arr.min()),
        "T_max": int(obs_arr.max()),
        "K": K,
        "pct_users_T_ge_10": float(np.mean(obs_arr >= 10) * 100),
        "repeat_rate": float(repeat_count / max(total_obs_for_repeat, 1)),
        "unique_choice_frac_median": float(np.median(unique_fracs)) if unique_fracs else 0.0,
        # Budget-specific
        "spend_per_obs_median": float(np.median(spend_arr)),
        "spend_per_obs_std": float(np.std(spend_arr)),
        "price_p5": float(np.percentile(price_arr, 5)),
        "price_p95": float(np.percentile(price_arr, 95)),
        "active_K_median": float(np.median(active_k_per_obs)),
        "active_K_over_total_K": float(np.median(active_k_per_obs) / max(K, 1)),
        "hhi_median": float(np.median(hhis)) if hhis else 0.0,
        "top_category_share_median": float(np.median(top_shares)) if top_shares else 0.0,
        "spend_cv_median": float(np.median(spend_cvs)) if spend_cvs else 0.0,
        "zero_spend_rate": float(zero_cells / max(total_cells, 1)),
    }


# ---------------------------------------------------------------------------
# Menu EDA (menus × choices)
# ---------------------------------------------------------------------------

def compute_menu_eda(
    train_logs: dict,
) -> dict:
    """Standardized EDA for menu-choice datasets.

    Args:
        train_logs: Dict mapping user_id -> MenuChoiceLog.

    Returns:
        Flat dict of EDA metrics.
    """
    n_users = len(train_logs)
    if n_users == 0:
        return {"n_users": 0}

    obs_per_user = []
    all_menu_sizes = []
    unique_items_per_user = []
    concentrations = []
    switch_rates = []
    entropies = []
    singleton_count = 0
    total_obs = 0
    repeat_count = 0
    total_obs_for_repeat = 0
    unique_fracs = []

    # For menu overlap: sample up to 200 users to keep it fast
    overlap_samples = []

    for uid, log in train_logs.items():
        T = len(log.choices)
        obs_per_user.append(T)
        total_obs += T

        # Menu sizes
        sizes = [len(m) for m in log.menus]
        all_menu_sizes.extend(sizes)
        singleton_count += sum(1 for s in sizes if s <= 1)

        # Unique items
        unique_items_per_user.append(log.num_items)

        # Choice concentration (modal-item share)
        if T > 0:
            counts = Counter(log.choices)
            modal = counts.most_common(1)[0][1]
            concentrations.append(modal / T)

        # Switch rate
        if T > 1:
            switches = sum(
                1 for i in range(1, T) if log.choices[i] != log.choices[i - 1]
            )
            switch_rates.append(switches / (T - 1))

        # Repeat rate: how often the chosen item was chosen before
        if T > 1:
            seen = set()
            for c in log.choices:
                if c in seen:
                    repeat_count += 1
                seen.add(c)
                total_obs_for_repeat += 1

        # Unique choice fraction
        if T > 0:
            unique_fracs.append(len(set(log.choices)) / T)

        # Choice entropy (normalized Shannon)
        if T > 0:
            counts = Counter(log.choices)
            probs = np.array(list(counts.values())) / T
            ent = float(-np.sum(probs * np.log2(probs + 1e-12)))
            max_ent = log2(max(len(counts), 2))
            entropies.append(ent / max_ent if max_ent > 0 else 0.0)

        # Menu overlap (sample)
        if len(overlap_samples) < 200:
            overlap_samples.append(log.menus)

    # Menu overlap rate: fraction of consecutive obs pairs sharing ≥1 item
    overlap_hits = 0
    overlap_total = 0
    for menus in overlap_samples:
        for i in range(1, len(menus)):
            if menus[i] & menus[i - 1]:
                overlap_hits += 1
            overlap_total += 1

    obs_arr = np.array(obs_per_user)
    menu_arr = np.array(all_menu_sizes)

    return {
        "type": "menu",
        "n_users": n_users,
        "total_obs": total_obs,
        "T_median": float(np.median(obs_arr)),
        "T_p25": float(np.percentile(obs_arr, 25)),
        "T_p75": float(np.percentile(obs_arr, 75)),
        "T_min": int(obs_arr.min()),
        "T_max": int(obs_arr.max()),
        "K": int(np.median(unique_items_per_user)),  # median unique items as K proxy
        "pct_users_T_ge_10": float(np.mean(obs_arr >= 10) * 100),
        "repeat_rate": float(repeat_count / max(total_obs_for_repeat, 1)),
        "unique_choice_frac_median": float(np.median(unique_fracs)) if unique_fracs else 0.0,
        # Menu-specific
        "menu_size_median": float(np.median(menu_arr)),
        "menu_size_p25": float(np.percentile(menu_arr, 25)),
        "menu_size_p75": float(np.percentile(menu_arr, 75)),
        "menu_size_min": int(menu_arr.min()),
        "menu_size_max": int(menu_arr.max()),
        "menu_overlap_rate": float(overlap_hits / max(overlap_total, 1)),
        "unique_items_per_user_median": float(np.median(unique_items_per_user)),
        "choice_concentration_median": float(np.median(concentrations)) if concentrations else 0.0,
        "switch_rate_median": float(np.median(switch_rates)) if switch_rates else 0.0,
        "choice_entropy_median": float(np.median(entropies)) if entropies else 0.0,
        "pct_singleton_menus": float(singleton_count / max(total_obs, 1) * 100),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_eda_summary(eda_list: list[dict]) -> None:
    """Print a compact cross-dataset EDA comparison table."""
    if not eda_list:
        return

    print("\n" + "=" * 130)
    print(" DATASET EDA SUMMARY")
    print("=" * 130)

    # Common metrics
    print(
        f"\n  {'Dataset':<20} {'Type':<7} {'N':>7} {'Total':>8} "
        f"{'T med':>6} {'T p25':>6} {'T p75':>6} "
        f"{'K':>5} {'%T≥10':>6} {'Repeat':>7} {'Uniq%':>6}"
    )
    print("  " + "-" * 125)
    for e in eda_list:
        ds = e.get("dataset", "?")
        print(
            f"  {ds:<20} {e['type']:<7} {e['n_users']:>7,} {e['total_obs']:>8,} "
            f"{e['T_median']:>6.0f} {e['T_p25']:>6.0f} {e['T_p75']:>6.0f} "
            f"{e['K']:>5} {e['pct_users_T_ge_10']:>5.0f}% {e['repeat_rate']:>6.1%} {e['unique_choice_frac_median']:>5.1%}"
        )
    print("  " + "-" * 125)

    # Budget-specific
    budgets = [e for e in eda_list if e["type"] == "budget"]
    if budgets:
        print(
            f"\n  BUDGET DETAILS"
            f"\n  {'Dataset':<20} {'Spend med':>10} {'Spend std':>10} "
            f"{'Price p5':>9} {'Price p95':>10} {'Active K':>9} {'K ratio':>8} "
            f"{'HHI':>6} {'Top share':>10} {'CV':>6} {'Zero%':>6}"
        )
        print("  " + "-" * 120)
        for e in budgets:
            print(
                f"  {e.get('dataset','?'):<20} {e['spend_per_obs_median']:>10.2f} {e['spend_per_obs_std']:>10.2f} "
                f"{e['price_p5']:>9.3f} {e['price_p95']:>10.3f} {e['active_K_median']:>9.1f} {e['active_K_over_total_K']:>7.1%} "
                f"{e['hhi_median']:>6.3f} {e['top_category_share_median']:>9.1%} {e['spend_cv_median']:>6.2f} {e['zero_spend_rate']:>5.1%}"
            )
        print("  " + "-" * 120)

    # Menu-specific
    menus = [e for e in eda_list if e["type"] == "menu"]
    if menus:
        print(
            f"\n  MENU DETAILS"
            f"\n  {'Dataset':<20} {'Menu med':>9} {'Menu p25':>9} {'Menu p75':>9} "
            f"{'Overlap':>8} {'Items/u':>8} {'Conc':>6} {'Switch':>7} {'Entropy':>8} {'Single%':>8}"
        )
        print("  " + "-" * 110)
        for e in menus:
            print(
                f"  {e.get('dataset','?'):<20} {e['menu_size_median']:>9.1f} {e['menu_size_p25']:>9.1f} {e['menu_size_p75']:>9.1f} "
                f"{e['menu_overlap_rate']:>7.1%} {e['unique_items_per_user_median']:>8.0f} "
                f"{e['choice_concentration_median']:>5.1%} {e['switch_rate_median']:>6.1%} "
                f"{e['choice_entropy_median']:>8.3f} {e['pct_singleton_menus']:>7.1f}%"
            )
        print("  " + "-" * 110)


# ---------------------------------------------------------------------------
# Feature correlation analysis
# ---------------------------------------------------------------------------

# Known RP feature prefixes/names (from Engine + extended extraction)
_RP_PREFIXES = (
    "garp", "ccei", "mpi", "harp", "hm_ratio", "hm_consistent", "hm_total",
    "vei", "n_scc", "scc_", "n_violations", "violation_", "pref_graph",
    "strict_pref", "transitivity", "util_", "lambda_", "mpi_", "cycle_",
    "sarp_", "warp_", "is_sarp", "is_warp", "is_garp", "is_harp",
    "menu_transitivity", "menu_pref", "menu_n_", "menu_max_", "menu_util",
    "choice_entropy", "choice_reversal", "is_congruent", "n_maximality",
    "max_scc", "n_warp", "n_sarp",
)
_RP_EXACT = {
    "choice_entropy", "choice_entropy_norm",
    "choice_reversal_count", "choice_reversal_ratio",
}


def _is_rp_feature(name: str) -> bool:
    if name in _RP_EXACT:
        return True
    return any(name.startswith(p) for p in _RP_PREFIXES)


def compute_correlation_summary(
    X_rp, X_base,
) -> dict:
    """Compute within-group and cross-group correlation statistics.

    Args:
        X_rp: DataFrame of RP features (users × features).
        X_base: DataFrame of baseline features (users × features).

    Returns:
        Dict with correlation summary: means, top pairs, redundant features.
    """
    import pandas as pd

    rp_cols = list(X_rp.columns)
    base_cols = list(X_base.columns)

    combined = pd.concat([X_base, X_rp], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    def _upper_triangle(corr_matrix):
        vals = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        return vals[~np.isnan(vals)]

    result = {
        "n_rp_features": len(rp_cols),
        "n_base_features": len(base_cols),
    }

    # Within-RP
    if len(rp_cols) >= 2:
        rp_corr = combined[rp_cols].corr()
        rp_vals = _upper_triangle(rp_corr)
        result["within_rp_mean_abs_corr"] = float(np.mean(np.abs(rp_vals)))
        result["within_rp_median_abs_corr"] = float(np.median(np.abs(rp_vals)))

        # Top RP-RP pairs
        rp_pairs = []
        for i, r1 in enumerate(rp_cols):
            for r2 in rp_cols[i + 1:]:
                c = combined[r1].corr(combined[r2])
                if not np.isnan(c):
                    rp_pairs.append({"f1": r1, "f2": r2, "r": round(float(c), 3)})
        rp_pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
        result["top_rp_rp_pairs"] = rp_pairs[:10]
        result["redundant_rp_pairs"] = [p for p in rp_pairs if abs(p["r"]) > 0.95]

    # Within-Base
    if len(base_cols) >= 2:
        base_corr = combined[base_cols].corr()
        base_vals = _upper_triangle(base_corr)
        result["within_base_mean_abs_corr"] = float(np.mean(np.abs(base_vals)))
        result["within_base_median_abs_corr"] = float(np.median(np.abs(base_vals)))

    # Cross-group (RP × Base)
    cross_pairs = []
    for rp in rp_cols:
        for b in base_cols:
            c = combined[rp].corr(combined[b])
            if not np.isnan(c):
                cross_pairs.append({"rp": rp, "base": b, "r": round(float(c), 3)})

    cross_vals = [abs(p["r"]) for p in cross_pairs]
    if cross_vals:
        result["cross_mean_abs_corr"] = float(np.mean(cross_vals))
        result["cross_median_abs_corr"] = float(np.median(cross_vals))

    cross_pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    result["top_cross_pairs"] = cross_pairs[:10]

    return result


def print_correlation_summary(corr: dict, dataset_name: str) -> None:
    """Print a compact correlation summary for one dataset."""
    print(f"\n  {dataset_name}: {corr['n_rp_features']} RP × {corr['n_base_features']} Base features")
    print(f"    RP-RP   mean |r| = {corr.get('within_rp_mean_abs_corr', 0):.3f}  "
          f"median |r| = {corr.get('within_rp_median_abs_corr', 0):.3f}")
    print(f"    Base-Base mean |r| = {corr.get('within_base_mean_abs_corr', 0):.3f}  "
          f"median |r| = {corr.get('within_base_median_abs_corr', 0):.3f}")
    print(f"    RP-Base  mean |r| = {corr.get('cross_mean_abs_corr', 0):.3f}  "
          f"median |r| = {corr.get('cross_median_abs_corr', 0):.3f}")

    redundant = corr.get("redundant_rp_pairs", [])
    if redundant:
        print(f"    Redundant RP pairs (|r| > 0.95): {len(redundant)}")
        for p in redundant[:5]:
            print(f"      {p['f1']:<30} ~ {p['f2']:<25} r={p['r']:+.3f}")

    top_cross = corr.get("top_cross_pairs", [])
    if top_cross:
        print(f"    Top RP-Base correlations:")
        for p in top_cross[:5]:
            print(f"      {p['rp']:<30} ~ {p['base']:<25} r={p['r']:+.3f}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_eda(eda_list: list[dict], output_dir: Path) -> None:
    """Save per-dataset EDA JSON files and a combined summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for e in eda_list:
        slug = _slugify(e.get("dataset", "unknown"))
        path = output_dir / f"eda_{slug}.json"
        with open(path, "w") as f:
            json.dump(e, f, indent=2)

    # Combined
    with open(output_dir / "eda_summary.json", "w") as f:
        json.dump(eda_list, f, indent=2)
    print(f"  EDA saved to {output_dir}/eda_summary.json ({len(eda_list)} datasets)")
