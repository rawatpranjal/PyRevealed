"""Feature extraction for ML benchmarks.

Three feature sets per user:
  - Baseline: RFM + spending/engagement stats (no RP library needed)
  - RP (Engine): Revealed preference metrics from Engine batch API
  - RP (Deep): Auditor scores, Encoder features, rolling-window consistency

All features are per-user scalars suitable for tabular ML.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pyrevealed.engine import Engine, results_to_dataframe


# ---------------------------------------------------------------------------
# Budget baseline features (from raw prices/quantities)
# ---------------------------------------------------------------------------

def extract_budget_baseline(
    users: list[tuple[np.ndarray, np.ndarray]],
    user_ids: list[str],
) -> pd.DataFrame:
    """Extract standard RFM + spending features from budget data."""
    records = []
    for (prices, quantities), uid in zip(users, user_ids):
        T, K = quantities.shape
        spend_per_obs = np.sum(prices * quantities, axis=1)
        total_qty_per_obs = np.sum(quantities, axis=1)

        total_spend = float(np.sum(spend_per_obs))
        mean_spend = float(np.mean(spend_per_obs))
        std_spend = float(np.std(spend_per_obs)) if T > 1 else 0.0
        max_spend = float(np.max(spend_per_obs))
        min_spend = float(np.min(spend_per_obs))

        n_obs = T
        mean_basket_size = float(np.mean(total_qty_per_obs))

        # Category concentration
        total_qty_per_cat = np.sum(quantities, axis=0)
        total_all = np.sum(total_qty_per_cat)
        if total_all > 0:
            shares = total_qty_per_cat / total_all
            herfindahl = float(np.sum(shares ** 2))
            top_share = float(np.max(shares))
            n_active_cats = int(np.sum(shares > 0))
        else:
            herfindahl = 1.0
            top_share = 1.0
            n_active_cats = 0

        # Temporal trend
        if T >= 3:
            x = np.arange(T, dtype=float)
            x_centered = x - x.mean()
            denom = np.sum(x_centered ** 2)
            spend_slope = float(np.sum(x_centered * (spend_per_obs - spend_per_obs.mean())) / denom) if denom > 0 else 0.0
        else:
            spend_slope = 0.0

        cv = std_spend / mean_spend if mean_spend > 0 else 0.0

        # Inter-observation variability
        if T >= 2:
            spend_diffs = np.diff(spend_per_obs)
            mean_abs_change = float(np.mean(np.abs(spend_diffs)))
        else:
            mean_abs_change = 0.0

        records.append({
            "user_id": uid,
            "n_obs": n_obs,
            "total_spend": total_spend,
            "mean_spend": mean_spend,
            "std_spend": std_spend,
            "max_spend": max_spend,
            "min_spend": min_spend,
            "mean_basket_size": mean_basket_size,
            "herfindahl": herfindahl,
            "top_category_share": top_share,
            "n_active_categories": n_active_cats,
            "spend_slope": spend_slope,
            "spend_cv": cv,
            "mean_abs_spend_change": mean_abs_change,
        })

    return pd.DataFrame(records).set_index("user_id")


# ---------------------------------------------------------------------------
# Budget RP features — Engine batch API
# ---------------------------------------------------------------------------

def extract_budget_rp(
    users: list[tuple[np.ndarray, np.ndarray]],
    user_ids: list[str],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Extract revealed preference features via Engine batch API."""
    if metrics is None:
        metrics = ["garp", "ccei", "mpi", "harp", "hm", "vei"]

    engine = Engine(metrics=metrics)
    results = engine.analyze_arrays(users)
    df = results_to_dataframe(results, user_ids=user_ids)

    # Derived features
    df["hm_ratio"] = df["hm_consistent"] / df["hm_total"].replace(0, 1)
    n_obs = np.array([u[0].shape[0] for u in users])
    df["violation_density"] = df["n_violations"] / np.maximum(n_obs * (n_obs - 1), 1)
    df["scc_ratio"] = df["max_scc"] / np.maximum(n_obs, 1)

    # Drop raw counts, keep ratios
    df = df.drop(columns=["hm_consistent", "hm_total", "compute_time_us"], errors="ignore")

    for col in ["is_garp", "is_harp", "utility_success"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


# ---------------------------------------------------------------------------
# Budget RP deep features — Auditor + Encoder + Rolling-window
# ---------------------------------------------------------------------------

def extract_budget_rp_deep(
    users: list[tuple[np.ndarray, np.ndarray]],
    user_ids: list[str],
    n_windows: int = 3,
) -> pd.DataFrame:
    """Extract deep RP features: Auditor, Encoder, rolling-window consistency.

    These are per-user features computed individually (not batch).
    More expensive but captures richer preference structure.

    Args:
        users: List of (prices, quantities) tuples.
        user_ids: Matching user IDs.
        n_windows: Number of rolling windows for temporal consistency.
    """
    from pyrevealed import BehaviorLog, BehavioralAuditor, PreferenceEncoder

    auditor = BehavioralAuditor()
    records = []

    for (prices, quantities), uid in zip(users, user_ids):
        T = prices.shape[0]
        row = {"user_id": uid}

        log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

        # --- Auditor features ---
        try:
            report = auditor.full_audit(log)
            row["audit_integrity"] = report.integrity_score
            row["audit_confusion"] = report.confusion_score
            row["audit_consistent"] = int(report.is_consistent)
        except Exception:
            row["audit_integrity"] = np.nan
            row["audit_confusion"] = np.nan
            row["audit_consistent"] = np.nan

        # --- Encoder features ---
        try:
            encoder = PreferenceEncoder()
            encoder.fit(log)
            if encoder.is_fitted:
                latent = encoder.extract_latent_values()
                marginal = encoder.extract_marginal_weights()
                row["enc_mean_latent"] = float(np.mean(latent))
                row["enc_std_latent"] = float(np.std(latent))
                row["enc_mean_marginal"] = float(encoder.mean_marginal_weight or 0)
                row["enc_latent_range"] = float(np.max(latent) - np.min(latent))
                row["enc_fitted"] = 1
            else:
                row["enc_mean_latent"] = 0.0
                row["enc_std_latent"] = 0.0
                row["enc_mean_marginal"] = 0.0
                row["enc_latent_range"] = 0.0
                row["enc_fitted"] = 0
        except Exception:
            row["enc_mean_latent"] = 0.0
            row["enc_std_latent"] = 0.0
            row["enc_mean_marginal"] = 0.0
            row["enc_latent_range"] = 0.0
            row["enc_fitted"] = 0

        # --- Rolling-window consistency ---
        if T >= 6 and n_windows >= 2:
            window_size = T // n_windows
            window_cceis = []
            for w in range(n_windows):
                start = w * window_size
                end = start + window_size if w < n_windows - 1 else T
                if end - start < 3:
                    continue
                try:
                    window_log = BehaviorLog(
                        cost_vectors=prices[start:end],
                        action_vectors=quantities[start:end],
                    )
                    from pyrevealed import compute_integrity_score
                    result = compute_integrity_score(window_log, tolerance=1e-4)
                    window_cceis.append(result.efficiency_index)
                except Exception:
                    pass

            if len(window_cceis) >= 2:
                row["rw_ccei_mean"] = float(np.mean(window_cceis))
                row["rw_ccei_std"] = float(np.std(window_cceis))
                row["rw_ccei_trend"] = float(window_cceis[-1] - window_cceis[0])
                row["rw_ccei_min"] = float(np.min(window_cceis))
            else:
                row["rw_ccei_mean"] = np.nan
                row["rw_ccei_std"] = np.nan
                row["rw_ccei_trend"] = np.nan
                row["rw_ccei_min"] = np.nan
        else:
            row["rw_ccei_mean"] = np.nan
            row["rw_ccei_std"] = np.nan
            row["rw_ccei_trend"] = np.nan
            row["rw_ccei_min"] = np.nan

        records.append(row)

    return pd.DataFrame(records).set_index("user_id")


# ---------------------------------------------------------------------------
# Menu baseline features
# ---------------------------------------------------------------------------

def extract_menu_baseline(
    user_logs: dict[str, object],
) -> pd.DataFrame:
    """Extract standard engagement features from menu-choice data."""
    records = []
    for uid, log in user_logs.items():
        n_sessions = len(log.choices)
        menu_sizes = [len(m) for m in log.menus]
        n_unique_items = len(log.all_items)

        # Choice concentration
        from collections import Counter
        choice_counts = Counter(log.choices)
        if choice_counts:
            max_choice_freq = max(choice_counts.values()) / n_sessions
            n_unique_choices = len(choice_counts)
        else:
            max_choice_freq = 0
            n_unique_choices = 0

        records.append({
            "user_id": uid,
            "n_sessions": n_sessions,
            "mean_menu_size": float(np.mean(menu_sizes)),
            "std_menu_size": float(np.std(menu_sizes)) if n_sessions > 1 else 0.0,
            "max_menu_size": max(menu_sizes),
            "min_menu_size": min(menu_sizes),
            "n_unique_items": n_unique_items,
            "items_per_session": n_unique_items / max(n_sessions, 1),
            "n_unique_choices": n_unique_choices,
            "max_choice_freq": max_choice_freq,
            "choice_concentration": n_unique_choices / max(n_unique_items, 1),
        })

    return pd.DataFrame(records).set_index("user_id")


# ---------------------------------------------------------------------------
# Menu RP features — Engine batch API
# ---------------------------------------------------------------------------

def extract_menu_rp(
    user_logs: dict[str, object],
) -> pd.DataFrame:
    """Extract revealed preference features for menu-choice data via Engine."""
    user_ids = list(user_logs.keys())
    engine_tuples = []
    for uid in user_ids:
        log = user_logs[uid]
        engine_tuples.append(log.to_engine_tuple())

    engine = Engine()
    results = engine.analyze_menus(engine_tuples)
    df = results_to_dataframe(results, user_ids=user_ids)

    df["hm_ratio"] = df["hm_consistent"] / df["hm_total"].replace(0, 1)
    n_obs = np.array([len(user_logs[uid].choices) for uid in user_ids])
    df["sarp_violation_density"] = df["n_sarp_violations"] / np.maximum(n_obs * (n_obs - 1) / 2, 1)
    df["warp_violation_density"] = df["n_warp_violations"] / np.maximum(n_obs * (n_obs - 1) / 2, 1)
    df["scc_ratio"] = df["max_scc"] / np.maximum(
        np.array([user_logs[uid].num_items for uid in user_ids]), 1
    )

    df = df.drop(columns=["hm_consistent", "hm_total", "compute_time_us"], errors="ignore")

    for col in ["is_sarp", "is_warp", "is_warp_la"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df
