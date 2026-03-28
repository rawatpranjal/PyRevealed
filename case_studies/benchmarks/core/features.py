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

from prefgraph.engine import Engine, results_to_dataframe


# ---------------------------------------------------------------------------
# Budget baseline features (from raw prices/quantities)
# ---------------------------------------------------------------------------

CORE_10_FEATURES = [
    "n_obs", "total_spend", "mean_spend", "std_spend",
    "mean_basket_size", "n_active_categories", "herfindahl",
    "top_category_share", "spend_slope", "spend_cv",
]


def extract_budget_baseline(
    users: list[tuple[np.ndarray, np.ndarray]],
    user_ids: list[str],
    feature_set: str = "full",
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

        # Recency and frequency (proxy — observation index as time)
        active_periods = spend_per_obs > 0
        n_active = int(np.sum(active_periods))
        purchase_rate = n_active / T  # true frequency rate
        if n_active > 0:
            last_active_idx = int(np.max(np.where(active_periods)[0]))
            recency = (T - 1 - last_active_idx) / max(T - 1, 1)  # 0=just bought, 1=bought only at start
        else:
            recency = 1.0
        # Inter-purchase gaps
        if n_active >= 2:
            active_indices = np.where(active_periods)[0]
            gaps = np.diff(active_indices)
            mean_gap = float(np.mean(gaps))
            std_gap = float(np.std(gaps))
        else:
            mean_gap = float(T)
            std_gap = 0.0

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
            "recency": recency,
            "purchase_rate": purchase_rate,
            "mean_inter_purchase_gap": mean_gap,
            "std_inter_purchase_gap": std_gap,
        })

    df = pd.DataFrame(records).set_index("user_id")
    if feature_set == "core":
        df = df[CORE_10_FEATURES]
    return df


# ---------------------------------------------------------------------------
# Budget RP features — Engine batch API
# ---------------------------------------------------------------------------

def extract_budget_rp(
    users: list[tuple[np.ndarray, np.ndarray]],
    user_ids: list[str],
    metrics: list[str] | None = None,
    extended: bool = True,
) -> pd.DataFrame:
    """Extract revealed preference features via Engine batch API.

    If extended=True, also runs per-user algorithm calls for ~30 additional
    features (distributional, graph, cycle, utility). Slower but richer.
    """
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
    df["vei_iqr"] = df["vei_q75"] - df["vei_q25"]
    df["scc_fragmentation"] = df["n_scc"] / np.maximum(n_obs, 1)

    # Drop raw counts, keep ratios
    df = df.drop(columns=["hm_consistent", "hm_total", "compute_time_us"], errors="ignore")

    for col in ["is_garp", "is_harp", "utility_success"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if extended:
        print(f"    Extracting extended RP features ({len(users)} users)...")
        df_ext = extract_budget_rp_extended(users, user_ids)
        df = pd.concat([df, df_ext], axis=1)

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
    from prefgraph import BehaviorLog, BehavioralAuditor, PreferenceEncoder

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
                    from prefgraph import compute_integrity_score
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
# Budget RP extended features — per-user algorithm calls (~30 new features)
# ---------------------------------------------------------------------------

def extract_budget_rp_extended(
    users: list[tuple[np.ndarray, np.ndarray]],
    user_ids: list[str],
) -> pd.DataFrame:
    """Extract extended RP features using per-user algorithm calls.

    Categories:
      - Distributional: VEI per-obs distribution, utility value distribution
      - Graph structural: cycle count, cycle lengths, SCC structure
      - Alternative scores: swaps index, observation contributions
      - Temporal: early vs late violation patterns
    """
    from prefgraph import BehaviorLog
    from prefgraph.algorithms.garp import check_garp
    from prefgraph.algorithms.mpi import compute_mpi
    from prefgraph.algorithms.vei import compute_vei
    from prefgraph.algorithms.utility import recover_utility

    records = []

    for (prices, quantities), uid in zip(users, user_ids):
        T = prices.shape[0]
        row = {"user_id": uid}
        log = BehaviorLog(cost_vectors=prices, action_vectors=quantities)

        # --- 1. VEI distributional features ---
        try:
            vei_result = compute_vei(log)
            ev = vei_result.efficiency_vector
            if ev is not None and len(ev) > 0:
                row["vei_std"] = float(np.std(ev))
                row["vei_q25"] = float(np.percentile(ev, 25))
                row["vei_q75"] = float(np.percentile(ev, 75))
                row["vei_iqr"] = row["vei_q75"] - row["vei_q25"]
                row["vei_below_90"] = int(np.sum(ev < 0.9))
                row["vei_below_90_frac"] = float(np.mean(ev < 0.9))
                # Temporal: first half vs second half
                mid = T // 2
                if mid > 0 and mid < T:
                    row["vei_first_half"] = float(np.mean(ev[:mid]))
                    row["vei_second_half"] = float(np.mean(ev[mid:]))
                    row["vei_temporal_shift"] = row["vei_second_half"] - row["vei_first_half"]
                else:
                    row["vei_first_half"] = np.nan
                    row["vei_second_half"] = np.nan
                    row["vei_temporal_shift"] = np.nan
            else:
                for k in ["vei_std", "vei_q25", "vei_q75", "vei_iqr",
                           "vei_below_90", "vei_below_90_frac",
                           "vei_first_half", "vei_second_half", "vei_temporal_shift"]:
                    row[k] = np.nan
        except Exception:
            for k in ["vei_std", "vei_q25", "vei_q75", "vei_iqr",
                       "vei_below_90", "vei_below_90_frac",
                       "vei_first_half", "vei_second_half", "vei_temporal_shift"]:
                row[k] = np.nan

        # --- 2. Utility recovery distributional features ---
        try:
            util_result = recover_utility(log)
            if util_result.success and util_result.utility_values is not None:
                uv = util_result.utility_values
                row["util_mean"] = float(np.mean(uv))
                row["util_std"] = float(np.std(uv))
                row["util_range"] = float(np.max(uv) - np.min(uv))
                row["util_cv"] = float(np.std(uv) / np.mean(uv)) if np.mean(uv) > 0 else 0.0
                # Gini coefficient of utility values
                uv_sorted = np.sort(uv)
                n = len(uv_sorted)
                idx = np.arange(1, n + 1)
                uv_sum = np.sum(uv_sorted)
                row["util_gini"] = float((2 * np.sum(idx * uv_sorted) / (n * uv_sum) - (n + 1) / n)) if uv_sum > 0 else 0.0

                if util_result.lagrange_multipliers is not None:
                    lm = util_result.lagrange_multipliers
                    row["lambda_mean"] = float(np.mean(lm))
                    row["lambda_std"] = float(np.std(lm))
                    row["lambda_cv"] = float(np.std(lm) / np.mean(lm)) if np.mean(lm) > 0 else 0.0
                else:
                    row["lambda_mean"] = np.nan
                    row["lambda_std"] = np.nan
                    row["lambda_cv"] = np.nan
            else:
                for k in ["util_mean", "util_std", "util_range", "util_cv", "util_gini",
                           "lambda_mean", "lambda_std", "lambda_cv"]:
                    row[k] = np.nan
        except Exception:
            for k in ["util_mean", "util_std", "util_range", "util_cv", "util_gini",
                       "lambda_mean", "lambda_std", "lambda_cv"]:
                row[k] = np.nan

        # --- 3. Graph structural features (from GARP) ---
        try:
            garp_result = check_garp(log)
            drp = garp_result.direct_revealed_preference
            tc = garp_result.transitive_closure
            srp = garp_result.strict_revealed_preference

            # Preference graph density
            n_pairs = T * (T - 1)
            row["pref_graph_density"] = float(np.sum(drp)) / max(n_pairs, 1)
            row["strict_pref_density"] = float(np.sum(srp)) / max(n_pairs, 1)
            row["transitivity_ratio"] = float(np.sum(tc)) / max(float(np.sum(drp)), 1)

            # Violation structure
            violations = garp_result.violations
            n_violations = len(violations)
            if n_violations > 0:
                cycle_lengths = [len(c) for c in violations]
                row["n_cycles"] = n_violations
                row["mean_cycle_length"] = float(np.mean(cycle_lengths))
                row["max_cycle_length"] = max(cycle_lengths)
                # Which observations appear in violations
                violating_obs = set()
                for cycle in violations:
                    violating_obs.update(cycle)
                row["violation_obs_frac"] = len(violating_obs) / T
                # Temporal: are violations early or late?
                if violating_obs:
                    obs_positions = np.array(list(violating_obs)) / max(T - 1, 1)
                    row["violation_mean_position"] = float(np.mean(obs_positions))
                else:
                    row["violation_mean_position"] = np.nan
            else:
                row["n_cycles"] = 0
                row["mean_cycle_length"] = 0.0
                row["max_cycle_length"] = 0
                row["violation_obs_frac"] = 0.0
                row["violation_mean_position"] = np.nan
        except Exception:
            for k in ["pref_graph_density", "strict_pref_density", "transitivity_ratio",
                       "n_cycles", "mean_cycle_length", "max_cycle_length",
                       "violation_obs_frac", "violation_mean_position"]:
                row[k] = np.nan

        # --- 4. MPI cycle cost distribution ---
        try:
            mpi_result = compute_mpi(log)
            if mpi_result.cycle_costs:
                costs = [c for _, c in mpi_result.cycle_costs]
                row["mpi_max_cycle_cost"] = float(max(costs))
                row["mpi_mean_cycle_cost"] = float(np.mean(costs))
                row["mpi_n_cycles"] = len(costs)
            else:
                row["mpi_max_cycle_cost"] = 0.0
                row["mpi_mean_cycle_cost"] = 0.0
                row["mpi_n_cycles"] = 0
        except Exception:
            row["mpi_max_cycle_cost"] = np.nan
            row["mpi_mean_cycle_cost"] = np.nan
            row["mpi_n_cycles"] = np.nan

        records.append(row)

    return pd.DataFrame(records).set_index("user_id")


# ---------------------------------------------------------------------------
# Menu RP extended features — per-user algorithm calls
# ---------------------------------------------------------------------------

def extract_menu_rp_extended(
    user_logs: dict[str, object],
) -> pd.DataFrame:
    """Extract extended RP features for menu-choice data.

    Categories:
      - Choice reversals: pairwise preference contradictions
      - Preference graph: density, transitivity
      - Congruence: full rationalizability test
      - Ordinal utility: preference ranking stats
    """
    from prefgraph.algorithms.abstract_choice import (
        validate_menu_sarp,
        validate_menu_warp,
        validate_menu_consistency,
        fit_menu_preferences,
    )
    from collections import Counter

    records = []

    for uid, log in user_logs.items():
        row = {"user_id": uid}
        T = len(log.choices)
        n_items = log.num_items

        # --- 1. Choice reversal features ---
        # Count pairwise reversals: A chosen over B in one menu, B over A in another
        pairwise_prefs: dict[tuple, int] = {}
        for menu, choice in zip(log.menus, log.choices):
            for item in menu:
                if item != choice:
                    pair = (choice, item)
                    pairwise_prefs[pair] = pairwise_prefs.get(pair, 0) + 1

        n_reversals = 0
        n_comparable_pairs = 0
        for (a, b), count in pairwise_prefs.items():
            reverse = (b, a)
            if reverse in pairwise_prefs:
                n_reversals += 1
                n_comparable_pairs += 1
            else:
                n_comparable_pairs += 1

        row["choice_reversal_count"] = n_reversals // 2  # Each reversal counted twice
        row["choice_reversal_ratio"] = (n_reversals // 2) / max(n_comparable_pairs, 1)

        # Choice entropy
        choice_counts = Counter(log.choices)
        total = sum(choice_counts.values())
        if total > 0:
            probs = np.array([c / total for c in choice_counts.values()])
            row["choice_entropy"] = float(-np.sum(probs * np.log2(probs + 1e-10)))
            row["choice_entropy_norm"] = row["choice_entropy"] / max(np.log2(n_items), 1e-10)
        else:
            row["choice_entropy"] = 0.0
            row["choice_entropy_norm"] = 0.0

        # --- 2. Preference graph features ---
        try:
            sarp_result = validate_menu_sarp(log)
            rp_matrix = sarp_result.revealed_preference_matrix
            tc_matrix = sarp_result.transitive_closure
            if rp_matrix is not None:
                n_pairs = n_items * (n_items - 1)
                row["menu_pref_density"] = float(np.sum(rp_matrix)) / max(n_pairs, 1)
                row["menu_transitivity"] = float(np.sum(tc_matrix)) / max(float(np.sum(rp_matrix)), 1) if np.sum(rp_matrix) > 0 else 1.0

                # Cycle structure from SARP violations
                violations = sarp_result.violations
                if violations:
                    cycle_lengths = [len(c) for c in violations]
                    row["menu_n_cycles"] = len(violations)
                    row["menu_max_cycle_len"] = max(cycle_lengths)
                else:
                    row["menu_n_cycles"] = 0
                    row["menu_max_cycle_len"] = 0
            else:
                row["menu_pref_density"] = np.nan
                row["menu_transitivity"] = np.nan
                row["menu_n_cycles"] = np.nan
                row["menu_max_cycle_len"] = np.nan
        except Exception:
            row["menu_pref_density"] = np.nan
            row["menu_transitivity"] = np.nan
            row["menu_n_cycles"] = np.nan
            row["menu_max_cycle_len"] = np.nan

        # --- 3. Congruence (full rationalizability) ---
        try:
            cong = validate_menu_consistency(log)
            row["is_congruent"] = int(cong.is_congruent)
            row["n_maximality_violations"] = cong.num_maximality_violations
        except Exception:
            row["is_congruent"] = np.nan
            row["n_maximality_violations"] = np.nan

        # --- 4. Ordinal utility / preference ranking ---
        try:
            pref_result = fit_menu_preferences(log)
            if pref_result.success and pref_result.utility_values is not None:
                uv = pref_result.utility_values
                row["menu_util_range"] = float(np.max(uv) - np.min(uv))
                row["menu_util_std"] = float(np.std(uv))
                row["menu_rank_complete"] = int(pref_result.is_complete)
            else:
                row["menu_util_range"] = np.nan
                row["menu_util_std"] = np.nan
                row["menu_rank_complete"] = 0
        except Exception:
            row["menu_util_range"] = np.nan
            row["menu_util_std"] = np.nan
            row["menu_rank_complete"] = 0

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
    extended: bool = True,
) -> pd.DataFrame:
    """Extract revealed preference features for menu-choice data via Engine.

    If extended=True, also runs per-user algorithm calls for additional
    features (reversals, preference graph, congruence, ordinal utility).
    """
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

    if extended:
        print(f"    Extracting extended menu RP features ({len(user_ids)} users)...")
        df_ext = extract_menu_rp_extended(user_logs)
        df = pd.concat([df, df_ext], axis=1)

    return df
